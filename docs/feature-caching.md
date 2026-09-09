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
