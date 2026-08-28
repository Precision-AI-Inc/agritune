# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository foundations: packaging, CI, pre-commit, structured logging, CLI entrypoint (`agritune --help`).
- Core schemas/protocols, agricultural dataset/manifest/split system, deterministic augmentation
  pipeline, encoder abstraction (fake + remote) with a rate-limited/retrying gateway, feature
  cache and resumable precomputation, `CachedFeatureProvider`, segmentation task (linear + TokenFPN
  decoders, losses, metrics), training engine (AMP, gradient accumulation, checkpointing/exact
  resume, provenance), optimizer/scheduler registries, and JSONL/TensorBoard/MLflow/W&B tracking.
- `agritune features verify/inspect/clean`, `agritune evaluate`, `agritune predict`, and
  `agritune encoder benchmark` commands, backed by their own services.
- `OnlineFeatureProvider` and `HybridFeatureProvider`, bridging an `EncoderGateway` into the
  synchronous `FeatureProvider` interface (the v0.2 online/hybrid feature path).
- Feature-space augmentation (`augmentations.feature`): patch dropout, token masking, Gaussian
  feature noise, CLS dropout, and channel dropout, wired into `Trainer` as an optional train-only
  step.
- Per-class precision/recall in `SegmentationMetric`; a configurable `checkpoint_top_k` for
  validation-ranked periodic checkpoints; a `temporal_split` dataset split strategy; a prediction
  visualization module (`--overlays` on `agritune predict`); `NeptuneTracker` and `CometTracker`.
- Agricultural domain-specific image augmentations: sensor channel dropout, ground-sample-distance
  (GSD) jitter, a red/green vegetation-index-balance jitter, and a seasonal/growth-stage hue shift.
- `augmentation.mode: hybrid` — deterministically reuses an offline variant most of the time, and
  derives a fresh online seed for a configurable fraction of `(sample, epoch, occurrence)` combinations.
- `PrefetchingFeatureProvider` (`precisionai.agritune.features.prefetch`): overlaps encoding with
  training via a bounded background queue, so an online/hybrid feature provider's network latency
  is hidden behind the GPU training step instead of blocking it. Cancellable mid-consumption (e.g.
  early stopping) without deadlocking `close()`.
- The API layer (`precisionai.agritune.api`): a thin FastAPI app (`create_app()`) with one route
  per CLI command (`dataset validate/inspect`, `features build/verify/inspect/clean`, `train`,
  `evaluate`, `predict`, `encoder benchmark`), calling the exact same `services/` functions as the
  CLI. `ValueError`/`FileNotFoundError`/`FeatureNotCachedError`/`CheckpointMismatchError` map to
  400/404/404/409 respectively; everything else surfaces as a 500, same as an uncaught CLI
  traceback. Encoder-backend selection was extracted into `services.encoder_selection`, shared by
  both entry points.
- Real Hydra config-group composition (`precisionai/agritune/configs/`): named YAML variants under
  `dataset/`, `encoder/`, `augmentation/`, `feature_provider/`, `task/`, `decoder/`, `optimizer/`,
  `scheduler/`, and `tracking/`, composed via a `defaults:` list in the packaged `config.yaml`.
  `cli.config.load_training_run_config` resolves either shape: a plain flat YAML file (unchanged,
  original behavior) or a `defaults:`-bearing file resolved through `hydra.compose()`, with the
  same `key=value` override list also able to swap group variants (`decoder=token_fpn`,
  `augmentation=online`). `feature_provider` (`cached`/`online`/`hybrid`), `augmentation`
  (`none`/`offline`/`online`/`hybrid`), and `tracking` selection are now wired end-to-end into
  `run_training`, which rejects the `augmentation=online|hybrid` + `feature_provider=cached`
  combination before training starts.
- Four new segmentation decoders, alongside a generalized MLP probe replacing the old
  linear-only baseline: `ASPPDecoder` (Atrous Spatial Pyramid Pooling, DeepLabv3-style),
  `PyramidPoolingDecoder` (PSPNet-style pyramid pooling), `SegmenterMaskTransformerDecoder`
  (Segmenter-style joint patch/class-token transformer with a scaled dot-product mask head), and
  `MaskFormerDecoder` (MaskFormer/Mask2Former-style query-based mask classification, combined into
  dense per-pixel scores rather than trained with the original papers' bipartite-matching loss).
  `LinearProbeDecoder` is replaced by `MLPProbeDecoder` (`decoder=mlp_probe`): with
  `hidden_dims=()` (the default) it is architecturally identical to the old linear probe;
  `hidden_dims=(256,)` etc. adds non-linear depth. `build_decoder` now accepts a generic
  `decoder_kwargs` passthrough forwarded to any decoder's constructor (also newly closing a gap
  for `TokenFPNDecoder`'s `cls_fusion`/`hidden_dim`, previously unconfigurable outside direct
  Python construction) — set via each decoder's own Hydra config-group variant
  (`configs/decoder/{mlp_probe,token_fpn,aspp,ppm,segmenter,mask_former}.yaml`).

### Changed

- The image augmentation pipeline (`augmentations.image`) now composes
  [albumentations](https://albumentations.ai/) transforms instead of hand-rolled PIL/numpy code,
  using `Compose.set_random_seed` for the same exact-reproducibility guarantee as before.

[Unreleased]: https://github.com/Precision-AI-Inc/agritune/compare/v0.1.0...HEAD
