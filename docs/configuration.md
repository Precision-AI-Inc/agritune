# Configuration

AgriTune composes configuration with Hydra/OmegaConf under `precisionai/agritune/configs/`:
`config.yaml` plus config groups (`dataset/`, `encoder/`, `augmentation/`, `feature_provider/`,
`task/`, `decoder/`, `optimizer/`, `scheduler/`, `tracking/`). The packaged `config.yaml`
composes:

```yaml
defaults:
  - dataset: agricultural_segmentation
  - encoder: fake
  - augmentation: none
  - feature_provider: cached
  - task: segmentation
  - decoder: mlp_probe
  - optimizer: adamw
  - scheduler: none
  - tracking: jsonl
  - _self_
```

Override group selections and plain fields from the CLI the same way Hydra always allows:

```bash
agritune train --config precisionai/agritune/configs/config.yaml \
    run_id=my-run \
    dataset.manifest_path=data/manifest.csv dataset.num_classes=5 \
    feature_store_dir=features \
    decoder=token_fpn optimizer=sgd scheduler=cosine scheduler.total_steps=10000 \
    augmentation=online feature_provider=online \
    tracking=local trainer.max_epochs=50
```

`augmentation=online`/`hybrid` combined with `feature_provider=cached` is rejected before training
starts — a cached provider's key would only ever match the first epoch's augmentation
fingerprint.

## Secrets and `.env` files

The `encoder=remote` group reads the API key from the environment rather than the config file:

```yaml
api_key: ${oc.env:AGRITUNE_ENCODER_API_KEY,null}
```

OmegaConf's `oc.env` resolver reads `os.environ` and nothing else — it has no `.env` support of its
own. AgriTune bridges the gap by loading a `.env` file into the environment *before* composition,
so the interpolation above resolves from a file with no change to any config:

```bash
pip install -e ".[dotenv]"      # python-dotenv is an optional extra
cp .env.example .env            # then fill in AGRITUNE_ENCODER_API_KEY
```

- The nearest `.env` at or above the working directory is used; set `AGRITUNE_ENV_FILE` to point
  somewhere else.
- **Variables already exported in your shell always win** — a `.env` file never overwrites them.
- `--api-key` on the CLI beats both.
- Without `python-dotenv` installed, a discovered `.env` is skipped with a warning (never
  silently); an explicitly requested one raises `ImportError`.
- `.env` is gitignored; `.env.example` is committed. Never put a key in a config file — resolved
  configs are written into `runs/<run_id>/` for reproducibility.

## Two ways to point ``--config`` at a file

- **A plain YAML file** (no `defaults:` key) is loaded directly and merged with the override list
  — the simplest way to fully spell out one run's configuration in a single file. This is what
  every example and test in this repository used before Hydra composition landed, and it still
  works unchanged.
- **A YAML file with a `defaults:` key** (like the packaged `config.yaml`) is instead resolved via
  genuine Hydra config-group composition — the override list can then both set plain fields
  (`trainer.max_epochs=50`) and swap group variants (`decoder=token_fpn`).

Point `--config` at your own copy of `config.yaml` (with different `defaults:` baked in, or
alongside your own `configs/dataset/your_dataset.yaml` variant) to reuse named presets across
runs instead of repeating an entire flat config file per experiment.

## Generating a starter flat config

`agritune config init --output my-run.yaml` writes a plain (non-Hydra) config file that spells out
every field `TrainingRunConfig` accepts — every field already at its default value, with the
handful of dataset-specific fields (`manifest_path`, `feature_store_dir`, `run_id`, `num_classes`)
marked `REQUIRED` as placeholders for you to fill in. It's the same shape as
`examples/segmentation/cwfid.yaml`, just without the CWFID-specific values. Pass `--force` to
overwrite an existing file at `--output`.

## Performance tuning

These are plain `TrainingRunConfig` fields, not a Hydra group — set them directly (flat config
file, or `key=value` overrides) alongside `batch_size`. None of them affect training math or
reproducibility, so they're excluded from the config fingerprint checkpoints are validated
against — changing any of them mid-run-series never invalidates a resume.

| Field | Default | Effect |
|---|---|---|
| `device` | `"cpu"` | Where the decoder and every batch's features/targets are moved before `forward`/`compute_loss` — `"cpu"`, `"cuda"`, or a specific GPU like `"cuda:3"`. Rejected up front (`ValueError`) if it names a CUDA device that is unavailable or out of range for the machine's GPU count — never a silent fallback to CPU. |
| `num_workers` | `0` | `DataLoader` worker subprocesses decoding/resizing raw images (or masks only, under `feature_provider: cached` — see [feature-caching.md](feature-caching.md#mask-only-loading-under-feature_provider-cached)). Raise this toward the machine's core count if "building batches" is slow relative to storage/decode speed. |
| `pin_memory` | `false` | Forwarded to the same `DataLoader`; speeds up the host-to-device copy of batch targets when `device` is a CUDA device. No effect on a CPU run. |
| `prefetch_factor` | `null` | Batches each `DataLoader` worker buffers ahead; requires `num_workers > 0`. Peak memory scales with `num_workers * prefetch_factor * batch_size` — cap it (e.g. `1`) if a high `num_workers` gets OOM-killed at a large `batch_size`. |
| `feature_read_workers` | `32` | Thread-pool size `CachedFeatureProvider` uses for concurrent store reads per batch — independent of `num_workers`, since these are I/O-bound threads (file open + safetensors deserialization) reading already-computed features, not raw images. Raise this toward the machine's core count when disk/store read latency, not CPU, is the batch-loading bottleneck (the common case for a large `store_type: sharded` store). |
| `store_type` | `"directory"` | `"directory"` (development scale, one file per sample) or `"sharded"` (production scale, many samples packed per shard file — see [feature-caching.md](feature-caching.md)). Must match whatever `feature_store_dir` was actually built as. |
| `entries_per_shard` | `1000` | Samples packed per shard file; only consulted when `store_type: sharded`. |

Every batch's feature fetch (a `FeatureProvider.get_features` call) also overlaps with the previous
batch's model compute automatically, via `PrefetchingFeatureLoader` — see
[feature-caching.md](feature-caching.md#overlapping-feature-fetch-with-compute). There's no config
flag for this; it's always on, for every `feature_provider` and every `device`.

## Group reference

| Group | Variants shipped | Selects |
|---|---|---|
| `dataset` | `agricultural_segmentation` | `manifest_path`, `num_classes` |
| `encoder` | `fake`, `remote` | `EncoderFingerprint` + `encoder_base_url`/`encoder_api_key` |
| `augmentation` | `none`, `offline`, `online`, `hybrid` | `AugmentationSelection` (mode, geometric/photometric transforms) |
| `feature_provider` | `cached`, `online`, `hybrid` | which `FeatureProvider` `run_training` builds |
| `task` | `segmentation` | loss config, `val_metric_name`, `higher_is_better` |
| `decoder` | `mlp_probe`, `token_fpn`, `aspp`, `ppm`, `segmenter`, `mask_former` | `decoder_name` + `decoder_kwargs` |
| `optimizer` | `adamw`, `adam`, `sgd` | `OptimizerConfig` |
| `scheduler` | `none`, `cosine`, `cosine_warmup`, `linear_warmup`, `polynomial`, `plateau` | `SchedulerConfig`, or no scheduler |
| `tracking` | `null`, `jsonl`, `tensorboard`, `local` (jsonl+tensorboard), `mlflow`, `wandb`, `neptune`, `comet` | `TrackingSelection` |

Every group's YAML lives in `precisionai/agritune/configs/<group>/<variant>.yaml` — copy one to
add a new named preset (e.g. a second `dataset/` variant for a different farm's manifest).
