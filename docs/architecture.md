# Architecture

AgriTune trains and evaluates agricultural segmentation decoders on frozen features from a
remote ViT encoder, with reproducible offline feature generation and optional rate-limit-aware
online feature extraction.

## Runtime flow

```text
Dataset
  │
  ├── image
  ├── segmentation mask
  └── metadata
       │
       ▼
Augmentation
       │
       ▼
FeatureProvider
  │
  ├── CachedFeatureProvider
  ├── OnlineFeatureProvider
  └── HybridFeatureProvider
       │
       ▼
EncoderFeatures
  │
  ├── patch_tokens [B, N, D]
  ├── cls_tokens   [B, D_cls]
  ├── patch_grid
  └── metadata
       │
       ▼
SegmentationTask
       │
       ▼
Decoder
       │
       ▼
Loss / Metrics
       │
       ▼
Trainer
```

## The rule that must never be broken

```text
Trainer and decoder must never directly call the encoder API.
```

Everything goes through `FeatureProvider`. This is what lets `features.provider` switch between
`cached` / `online` / `hybrid` in config with zero changes to the trainer or decoder, and what
lets the entire test suite and CI run without a live encoder API key (see
[encoder.md](encoder.md) and `FakeEncoderBackend`).

## Layer boundary this repo protects most aggressively

```text
          EncoderBackend
                │
                ▼
          EncoderGateway
                │
                ▼
FeatureStore ← FeatureProvider
                │
                ▼
              Task
                │
                ▼
             Trainer
```

If this separation stays clean, AgriTune can grow from an agricultural segmentation repo into a
general agricultural downstream-training framework (classification, regression, depth, detection)
without a rewrite of the training engine.

## Orchestration layering

```text
CLI ─────┐
         │
API ─────┼──> Services ───> core implementation (schemas/data/encoder/features/tasks/training/...)
         │
Python ──┘
```

`precisionai.agritune.services` is the single orchestration layer. The CLI (`precisionai.agritune.cli`)
and the thin FastAPI layer (`precisionai.agritune.api`) both call the same services — neither
duplicates business logic.

## API path containment

The CLI is a trusted local tool and accepts any path the invoking user can read or write. The API
is not: every request field that names a filesystem path (`manifest_path`, `store`,
`checkpoint_path`, `output_dir`, `config_path`, `augmentation_config_path`) is resolved through
`precisionai.agritune.api.paths.resolve_under_root` against the directory
`precisionai.agritune.api.config.get_api_root` returns (`AGRITUNE_API_ROOT`, defaulting to the
server's current working directory). An absolute path or a `..` segment that would resolve outside
that root is rejected with HTTP 400 before any file is touched. This API has no built-in
authentication — see `SECURITY.md` for what that means for deployment.

See also: [configuration.md](configuration.md), [datasets.md](datasets.md), [encoder.md](encoder.md),
[feature-caching.md](feature-caching.md), [augmentation.md](augmentation.md),
[segmentation.md](segmentation.md), [reproducibility.md](reproducibility.md).
