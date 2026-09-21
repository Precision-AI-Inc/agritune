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
