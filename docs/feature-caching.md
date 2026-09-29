# Feature caching

Feature cache keys fingerprint sample ID, image hash, augmentation configuration/seed, encoder
model, encoder revision, encoder preprocessing, and feature schema version — see
[encoder.md](encoder.md) for why the encoder revision component is best-effort rather than
authoritative. `DirectoryFeatureStore` is the development-scale store; `ShardedFeatureStore` is the
production store (buffered writes, auto-flushed by shard size). Both support `verify()` (checksum
comparison against `agritune features verify`) and `clean()` (orphaned tensor/meta file removal,
`agritune features clean`). Precomputation (`agritune features build`) is resumable — restarting
after a partial run only encodes the missing samples.

For cached training with `augmentation.mode: offline`, precompute the same deterministic variant
(and its unaugmented validation features) through the service API:

```python
from precisionai.agritune.augmentations.image import AugmentationMode, ImageAugmentationPipeline
from precisionai.agritune.services.feature_service import build_features

await build_features(
    "manifest.csv",
    store=store,
    encoder=encoder,
    encoder_fingerprint=fingerprint,
    augmentation_mode=AugmentationMode.OFFLINE,
    augmentation_pipeline=ImageAugmentationPipeline(pipeline_config),
    global_seed=training_seed,
    augmentation_variant=variant,
)
```

The pipeline, seed, variant, encoder fingerprint, and manifest must match training. Online and
hybrid augmentation are intentionally rejected by precomputation because they produce an unbounded
sequence; use an online or hybrid feature provider for those modes.

## Feature providers

Three `FeatureProvider` implementations (`precisionai.agritune.features.provider`), matching
`features.provider: cached / online / hybrid`:

- `CachedFeatureProvider` — reads precomputed features from a store; raises
  `FeatureNotCachedError` on a miss rather than silently falling back to the network.
- `OnlineFeatureProvider` — encodes every batch fresh through an `EncoderGateway`, bridging its
  `async` `encode()` to the synchronous `FeatureProvider` interface via `asyncio.run`.
- `HybridFeatureProvider` — reads a store first; on a miss, encodes through the gateway and writes
  the result back (write-through), so a hybrid run against a partially-precomputed store only
  ever encodes what it has not already seen.

`PrefetchingFeatureProvider` (`precisionai.agritune.features.prefetch`) wraps an online/hybrid
provider with a bounded background-thread queue, so the encoder's network latency is hidden behind
the GPU training step instead of blocking it — it must
be given the full, ordered sequence of upcoming batches up front, and `close()` cancels cleanly
even if consumption stopped early (e.g. early stopping).

## Overlapping feature fetch with compute

`PrefetchingFeatureLoader` (`precisionai.agritune.training.prefetch`) solves a related but
differently-shaped problem: `Trainer`'s training loop and `evaluate()` both need to fetch features
*and* read each batch's `targets`/`samples` in a single pass, so they can't hand
`PrefetchingFeatureProvider` the batch sequence up front without either double-iterating it or
materializing a whole epoch into memory. Instead, `PrefetchingFeatureLoader` wraps any
`FeatureProvider` (`cached`, `online`, or `hybrid` alike) plus a stream of `TrainingBatch`es and
runs exactly one batch ahead in a single background thread, yielding `(batch, features, targets)`
already moved to the target device. It's wired into `Trainer`/`evaluate()` unconditionally — there
is no config flag to disable it — so the model never blocks on a synchronous `get_features()` call
while the GPU sits idle. `feature_read_workers` (see [configuration.md](configuration.md#performance-tuning))
controls how much `CachedFeatureProvider` itself parallelizes the read this loader is prefetching
ahead of; the two settings are complementary, not redundant — one hides read latency behind
compute, the other shortens the read itself.

## Mask-only loading under `feature_provider: cached`

Under `feature_provider: cached`, a batch's `EncoderFeatures` come entirely from the store —
`CachedFeatureProvider` never touches `PreparedSample.image`. `build_training_batches`/
`build_static_augmented_batches` (`precisionai.agritune.services.segmentation_common`) accept a
`load_images` flag (computed automatically from `config.feature_provider` inside
`training_service._build_train_batches` — never set directly) that, when `False`, skips decoding
the raw image entirely and loads only the mask (`ManifestDataset.load_target`), computing the same
cache key `agritune features build` wrote under so the batch still reads the right entry.

This only works because `augmentation.mode: offline`'s recorded `AugmentationRecord` (from
`alb.Compose(..., save_applied_params=True)`) can be replayed against the mask alone:
`ImageAugmentationPipeline` splits geometric transforms (which must produce a pixel-identical view
on both the image and the mask) from photometric transforms (image-only, but still part of the
cache-key hash) into two independently-seeded composes, so `apply_to_mask` can replay the
geometric compose on the real mask and the photometric compose against a shape-matched dummy array
— reproducing the exact same resolved parameters `hash_augmentation` needs, without ever decoding
the real image. `feature_provider: online`/`hybrid` still decode every image, since those modes
call the encoder directly on it.

## Preloading features into memory

With `feature_provider: cached`, a linear-probe epoch is dominated by reading features, not by the
decoder. Streaming from a store, every batch pays one `store.read()` per sample, a fresh
`EncoderFeatures` (with validation) per sample, a concatenation, and a PNG decode per mask, on
every epoch. `feature_preload` moves all of that to a single pass before training:

- `InMemoryFeatureProvider` (`precisionai.agritune.features.memory`) reads the training and
  validation samples' features once through the store's `read_many()` (which loads each shard file
  once), packs them into one `(samples, max_patches, patch_dim)` tensor, checks it for non-finite
  values once, and serves each batch as a single `index_select`. Batches match
  `concatenate_encoder_features` exactly: tokens are padded to the largest patch count in the
  batch, and `valid_patch_mask` is only present when the batch mixes grid sizes.
- `PreloadedBatches` (`precisionai.agritune.services.segmentation_common`) walks the usual batches
  iterable once, keeps every `PreparedSample` (including its augmentation record, which the cache
  key depends on) and one stacked target tensor in the narrowest integer dtype that holds the
  labels (`uint8` for ordinary masks), and replays it every epoch. The loss and metric widen
  targets to `int64` on the device.

`feature_preload: host` keeps both buffers in RAM. Each batch is gathered into pinned memory and
copied to the GPU on `PrefetchingFeatureLoader`'s dedicated CUDA stream, overlapping the previous
step. `feature_preload: device` keeps them on the GPU, so no batch crosses PCIe.

Size the memory before choosing a mode:

| Buffer | Size |
|---|---|
| Features | `(train + val samples) x patches x patch_dim x bytes per value` |
| Masks | `(train + val samples) x mask height x mask width` bytes (`uint8`) |

For 151,291 samples with 384-dimensional tokens, a 16x16 grid in `float16` is about 30 GB (fits on
one 80 GB GPU with room to train), while a native 57x38 grid is about 250 GB in `float16` (host RAM
only). `feature_preload_dtype: float16` halves both size and copy traffic relative to the API's
`float32`; values that would overflow `float16` are rejected while preloading, with a hint to use
`bfloat16`.

Preloading only applies to deterministic batches (`augmentation.mode: none` or `offline`), which
`feature_provider: cached` already requires. With the data in memory, `shuffle: true` becomes
cheap; without preloading it is rejected, because shuffled reads from a sharded store reload a
whole shard for nearly every sample.
