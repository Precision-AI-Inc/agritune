# AgriTune tutorials

Task-oriented walkthroughs and a full configuration reference for the packaged examples, built on
the two public datasets shipped with this repo: [CWFID](https://github.com/cwfid/dataset) (small,
fast to iterate on) and [PhenoBench](https://www.phenobench.org) (larger, closer to production
scale). Every command below assumes both are prepared — see
[datasets/README.md](datasets/README.md) for the full details on each:

```bash
python examples/datasets/prepare_cwfid.py --output examples/datasets/cwfid --max-samples 24 --size 384
python examples/datasets/prepare_phenobench.py --output examples/datasets/phenobench --max-samples 40 --size 384
```

Most commands below default to the hosted `pai-embedding` encoder
(`--base-url https://embeddings.precision.ai/v1 --model pai-embedding`, API key read from
`AGRITUNE_ENCODER_API_KEY` — see the root [README](../README.md)'s Quick Start). No encoder
access? Swap `--base-url ... --model pai-embedding` for `--model fake-encoder` anywhere below (or
drop `encoder_base_url`/set `encoder_fingerprint.model: fake-encoder` in a YAML config) — every
command still runs, against synthetic features.

See also: [SANITY_CHECK.md](SANITY_CHECK.md) (a single linear CWFID walkthrough with the hosted
encoder, including what a run directory contains) and
[FEATURE_TEST_COMMANDS.md](FEATURE_TEST_COMMANDS.md) (an exhaustive command matrix across every
decoder/provider/optimizer/scheduler combination).

---

## Configuration reference

`agritune train --config <file.yaml> [overrides...]` accepts a **flat, non-Hydra** YAML file (the
shape every config in [examples/segmentation/](segmentation/) uses) resolving directly to a
`TrainingRunConfig` — see
[precisionai/agritune/services/training_service.py](../precisionai/agritune/services/training_service.py)
and [precisionai/agritune/cli/config.py](../precisionai/agritune/cli/config.py) for the
authoritative schema, and [docs/configuration.md](../docs/configuration.md) for the alternative
Hydra config-group composition (a YAML file with a `defaults:` key).

Every field below can also be set as a `key=value` / `key.nested=value` CLI override instead of
editing the file, e.g.:

```bash
agritune train --config examples/segmentation/cwfid.yaml trainer.max_epochs=10 optimizer.lr=1e-4
```

[examples/segmentation/cwfid.yaml](segmentation/cwfid.yaml) and
[phenobench.yaml](segmentation/phenobench.yaml) both spell out **every** field below, including
ones already equal to their default — copy whichever is closer to your use case. `agritune config
init --output my_config.yaml` writes the same fully-commented template
([precisionai/agritune/configs/templates/full_config.yaml](../precisionai/agritune/configs/templates/full_config.yaml)).

### Dataset / run identity

| Field | Default | Meaning |
|---|---|---|
| `manifest_path` | *required* | Dataset manifest CSV — see [datasets/README.md](datasets/README.md). |
| `feature_store_dir` | *required* | Feature store directory (built via `agritune features build`); read under `cached`/`hybrid`, write-through populated under `hybrid`, unused under `online`. |
| `run_root` | *required* | Directory `runs/<run_id>/` is created under. |
| `run_id` | *required* | Unique run identifier — becomes the `runs/<run_id>/` subdirectory name. |
| `num_classes` | *required* | Number of segmentation classes. |
| `seed` | `0` | Global seed; `trainer.seed` falls back to this when unset. |

### Encoder identity — `encoder_fingerprint`

| Field | Default | Meaning |
|---|---|---|
| `model` | *required* | Encoder model alias/identifier (e.g. `pai-embedding`, `fake-encoder`). |
| `revision` | `null` | Encoder revision, when known. `null` for the hosted API, which exposes none — see [docs/encoder.md](../docs/encoder.md). |
| `preprocessing` | `""` | Stable label for preprocessing params (resize, normalization, `native_resolution`), for cache invalidation. |

Must match whatever `feature_store_dir` was actually built with under `cached`/`hybrid` — every
lookup misses otherwise.

### Feature provider

| Field | Default | Meaning |
|---|---|---|
| `feature_provider` | `cached` | `cached` (reads only from `feature_store_dir`) \| `online` (calls the encoder every sample, every epoch) \| `hybrid` (reads the cache, write-through populates whatever is missing). |
| `store_type` | `directory` | `directory` (development scale, one file per sample) \| `sharded` (production scale, many samples packed per shard file) — must match how `feature_store_dir` was actually built. |
| `entries_per_shard` | `1000` | Samples packed per shard file; only consulted when `store_type: sharded`. |
| `encoder_base_url` | `null` | Hosted encoder API base URL; `null` uses `FakeEncoderBackend`. Ignored entirely under `cached`. |
| `encoder_api_key` | `null` | Ignored when `encoder_base_url` is `null`. Never hardcode a real key — use `${oc.env:AGRITUNE_ENCODER_API_KEY,null}`, as every packaged example does. |

### Image augmentation — `augmentation`

Full mechanics in [docs/augmentation.md](../docs/augmentation.md); see also the
[augmentation-mode](#tutorial-training-with-none--offline--online-augmentation) and
[resize-vs-augmentation](#tutorial-resize-only-normalization-vs-the-full-augmentation-pipeline)
deep dives below.

| Field | Default | Meaning |
|---|---|---|
| `mode` | `none` | `none` (no augmentation) \| `offline` (one fixed variant, reused every epoch) \| `online` (re-augmented fresh every epoch) \| `hybrid` (mostly offline, online for a configurable fraction). `online`/`hybrid` **require** `feature_provider: online`/`hybrid` — rejected up front otherwise. |
| `variant` | `0` | Offline variant index; ignored outside `offline`/`hybrid`. |
| `hybrid_online_probability` | `0.3` | Fraction of `(sample, epoch)` combinations that take the fresh-online path under `hybrid`; ignored otherwise. |

`geometric` (a `None`/`0` value disables that transform):

| Field | Default | Meaning |
|---|---|---|
| `resize` | `null` | Target `[width, height]`, applied first if set. The **one exception** to `mode`: applied to every training *and* validation batch regardless of mode (including `none`) — deterministic dimensional normalization, not augmentation, required whenever the dataset's images/masks don't already share one native size. |
| `random_crop` | `null` | `[width, height]` crop, applied after resize if set. May exceed the image's own size (padded, not raised). |
| `horizontal_flip_probability` | `0.0` | Probability in `[0, 1]`. |
| `vertical_flip_probability` | `0.0` | Probability in `[0, 1]`. |
| `rotation_max_degrees` | `0.0` | Maximum absolute rotation in degrees; `0` disables it. |

`photometric` (image-only; agricultural-domain additions noted):

| Field | Default | Meaning |
|---|---|---|
| `brightness_range` | `null` | `[min, max]` multiplicative factor around `1.0`. |
| `contrast_range` | `null` | `[min, max]` multiplicative factor around `1.0`. |
| `saturation_range` | `null` | `[min, max]` multiplicative factor around `1.0`. |
| `hue_max_degrees` | `0.0` | Maximum absolute hue shift in degrees (of 360). |
| `blur_probability` | `0.0` | Probability of Gaussian blur. |
| `blur_radius_range` | `[0.1, 2.0]` | Gaussian-sigma range sampled when blur applies. |
| `noise_probability` | `0.0` | Probability of Gaussian pixel noise. |
| `noise_std_range` | `[0.01, 0.05]` | Normalized-`[0,1]`-scale std range sampled when noise applies. |
| `channel_dropout_probability` | `0.0` | Probability of zeroing one RGB channel entirely — simulates a failed/missing sensor band. |
| `gsd_jitter_probability` | `0.0` | Probability of a downscale-then-upscale resolution-degrading pass — simulates variable ground-sample-distance across flights. |
| `gsd_jitter_scale_range` | `[0.5, 0.9]` | Downscale factor range (smaller = coarser). |
| `vegetation_index_jitter_probability` | `0.0` | Probability of an independent red/green channel jitter — an RGB-only proxy for NDVI sensitivity (no NIR band). |
| `vegetation_index_jitter_range` | `[-15.0, 15.0]` | Additive shift (0–255 scale), sampled independently for red and green. |
| `seasonal_color_shift_probability` | `0.0` | Probability of a seasonal/growth-stage hue shift. |
| `seasonal_hue_shift_degrees` | `0.0` | Maximum absolute hue shift, OpenCV 0–179 hue-circle units. |

### Feature-space augmentation — `feature_augmentation`

Applied to training batches' *features* only (after `feature_provider.get_features`, before the
decoder) — never to validation, never touches the feature cache/store, so it composes freely with
any `feature_provider` and any `augmentation.mode`. A `0.0` probability/std disables that
transform.

| Field | Default | Meaning |
|---|---|---|
| `patch_dropout_probability` | `0.0` | Drop probability per patch, in `[0, 1)`. |
| `token_masking_probability` | `0.0` | Mask probability per patch, in `[0, 1]`. |
| `token_mask_value` | `0.0` | Constant value masked patches are set to. |
| `gaussian_noise_std` | `0.0` | Std of additive Gaussian noise on patch/CLS tokens. |
| `cls_dropout_probability` | `0.0` | Drop probability per sample's CLS token, in `[0, 1)`; a no-op when a sample has none. |
| `channel_dropout_probability` | `0.0` | Drop probability per embedding channel, in `[0, 1)`. |

### Decoder — `decoder_name` / `decoder_kwargs`

| `decoder_name` | `decoder_kwargs` (all forwarded to the constructor) |
|---|---|
| `mlp_probe` (default) | `hidden_dims` (`[]` — empty = a single `nn.Linear(patch_dim, num_classes)`), `dropout` (`0.0`, applied after each hidden layer, never the last). No spatial context beyond one patch. |
| `token_fpn` | `cls_dim` (from the encoder; not user-set), `hidden_dim` (`128`), `num_layers` (`2`), `cls_fusion` (`none` \| `concat` \| `film`). Convolutional refinement across the patch grid. |
| `aspp` | `hidden_dim` (`128`), `atrous_rates` (`[6, 12, 18]`), `num_layers` (`1`). |
| `ppm` | `hidden_dim` (`128`), `pool_sizes` (`[1, 2, 3, 6]`), `num_layers` (`1`). |
| `segmenter` | `hidden_dim` (`192`), `num_layers` (`2`), `num_heads` (`3`), `mlp_ratio` (`4`), `dropout` (`0.0`). |
| `mask_former` | `hidden_dim` (`128`), `num_queries` (`32`), `num_layers` (`2`), `num_heads` (`4`), `mlp_ratio` (`4`), `dropout` (`0.0`). |

`patch_dim`/`num_classes`/`output_size` are always inferred at run time (from the feature store and
the augmented target's spatial size) — never set them in `decoder_kwargs`.

### Batching / device

| Field | Default | Meaning |
|---|---|---|
| `batch_size` | `4` | |
| `device` | `cpu` | `cpu` (always safe on CPU-only machines) \| `cuda` \| a specific GPU like `cuda:3`. Rejected up front if unavailable/out of range — never silently falls back to CPU. |
| `num_workers` | `0` | `DataLoader` worker subprocesses for decode/augment; `0` runs in the main process. A performance knob only — excluded from the run's reproducibility fingerprint. |
| `pin_memory` | `false` | Faster host→device copy; only helps on a CUDA device. |
| `prefetch_factor` | `null` | Batches each worker buffers ahead; `null` defers to `DataLoader`'s default (`2`); requires `num_workers > 0`. Peak memory scales with `num_workers * prefetch_factor * batch_size`. |
| `feature_read_workers` | `32` | Thread-pool size for `CachedFeatureProvider`'s concurrent store reads (only under `feature_provider: cached`); independent of `num_workers`. |
| `val_fraction` | `0.2` | Fraction of samples held out for validation (random split). Validation is always unaugmented, regardless of `augmentation.mode`. |

### Optimizer — `optimizer`

| Field | Default | Meaning |
|---|---|---|
| `name` | `adamw` | `adamw` \| `adam` \| `sgd`. |
| `lr` | `3e-4` | Learning rate. |
| `weight_decay` | `1e-2` | L2 penalty for `sgd`; decoupled for `adam`/`adamw`. |
| `betas` | `[0.9, 0.999]` | Adam/AdamW momentum coefficients; ignored for `sgd`. |
| `momentum` | `0.9` | SGD momentum; ignored for `adam`/`adamw`. |

### LR scheduler — `scheduler`

`enabled: false` (or an empty mapping) disables the scheduler entirely, equivalent to omitting the
whole section.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | *(section presence)* | `false` disables the scheduler regardless of the fields below. |
| `name` | `constant` | `constant` \| `cosine` \| `linear_warmup` \| `polynomial` \| `plateau`. |
| `warmup_steps` | `0` | Linear warmup duration, in optimizer steps; ignored by `constant`/`plateau`. |
| `total_steps` | `null` | Required by `cosine`/`linear_warmup`/`polynomial`. |
| `polynomial_power` | `1.0` | Decay exponent for `polynomial`. |
| `plateau_factor` | `0.5` | Multiplicative LR reduction `plateau` applies when triggered. |
| `plateau_patience` | `10` | Validation passes with no improvement before `plateau` reduces the LR. |

Every scheduler here steps per **optimizer step**, not per micro-batch.

### Trainer — `trainer`

| Field | Default | Meaning |
|---|---|---|
| `max_epochs` | *required* | Number of epochs to train for. Excluded from the config fingerprint — bumping it to resume for longer never invalidates an existing checkpoint. |
| `accumulation_steps` | `1` | Micro-batches accumulated before each optimizer step. |
| `grad_clip_norm` | `null` | Max gradient norm for clipping; `null` disables it. |
| `precision` | `fp32` | `fp32` (autocast disabled) \| `fp16` (autocast + gradient scaling) \| `bf16` (autocast, no scaling needed). |
| `seed` | *(falls back to top-level `seed`)* | Deterministic seed applied at the start of `fit()`. |
| `early_stopping_patience` | `null` | Stop after this many validation passes with no improvement; `null` disables it. |
| `checkpoint_every_n_steps` | `null` | Also write a periodic checkpoint every this many optimizer steps, in addition to end-of-epoch/best. |
| `strict_resume` | `true` | `false` logs a warning instead of raising `CheckpointMismatchError` on a fingerprint mismatch when resuming, and resumes anyway. Excluded from the config fingerprint itself. Only use after confirming a mismatch is benign — decoder/optimizer state still loads as-is, so a genuine architecture/dataset change resumed this way can silently corrupt the run. |

### Loss — `loss`

| Field | Default | Meaning |
|---|---|---|
| `name` | `ce` | `ce` \| `bce` \| `dice` \| `ce_dice` \| `bce_dice`. |
| `ignore_index` | `-100` | Pixel value excluded from every term. |
| `class_weights` | `null` | Per-class weights — `weight` for CE, `pos_weight` for BCE. |
| `ce_weight` | `1.0` | Weight applied to the CE/BCE term in a combined loss. |
| `dice_weight` | `1.0` | Weight applied to the Dice term in a combined loss. |

### Validation / checkpointing

| Field | Default | Meaning |
|---|---|---|
| `val_metric_name` | `mean_iou` | Metric key `Trainer` tracks for best-checkpoint/early-stopping/scheduler-plateau decisions. |
| `higher_is_better` | `true` | Set `false` for a loss-like metric. |
| `checkpoint_top_k` | `3` | Best-by-`val_metric_name` periodic checkpoints retained; `0` keeps every one. |

### Tracking — `tracking`

| Field | Default | Meaning |
|---|---|---|
| `backends` | `[jsonl]` | Any of `null`, `jsonl`, `tensorboard`, `mlflow`, `wandb`, `neptune`, `comet` — a list, so several can fan out at once. Only the settings for backends actually selected are consulted. |
| `mlflow_experiment_name` | `null` | |
| `mlflow_tracking_uri` | `null` | |
| `wandb_project` | `agritune` | |
| `neptune_project` | `null` | Required, as `"workspace/project"`, if `neptune` is selected. |
| `comet_project_name` | `agritune` | |

---

## Pipeline walkthrough

Every `agritune` subcommand, demonstrated against both CWFID and PhenoBench. All commands assume
the Quick Start dataset preparation above already ran.

### `dataset` — validate, inspect, scaffold

```bash
agritune dataset validate --manifest examples/datasets/cwfid/manifest.csv --num-classes 3
agritune dataset inspect --manifest examples/datasets/cwfid/manifest.csv

agritune dataset validate --manifest examples/datasets/phenobench/manifest.csv --num-classes 3
agritune dataset inspect --manifest examples/datasets/phenobench/manifest.csv
```

`validate` exits `1` and prints one line per issue (out-of-range mask labels, image/mask size
mismatches, missing files, ...) if anything is wrong; `inspect` prints a JSON summary (sample
count, class-pixel histograms, image size distribution). Starting a manifest from scratch instead
of using a prep script:

```bash
agritune dataset init --output my_dataset/manifest.csv
```

### `config init` — scaffold a training config

```bash
agritune config init --output my_training_config.yaml
```

Writes the same fully-commented template referenced throughout the
[Configuration reference](#configuration-reference) above
([precisionai/agritune/configs/templates/full_config.yaml](../precisionai/agritune/configs/templates/full_config.yaml))
— a fill-in-the-blanks starting point for a new dataset, instead of copying `cwfid.yaml`/
`phenobench.yaml`.

### `encoder benchmark` — throughput/latency sizing (optional)

Requires real encoder access (skip if you're only using `fake-encoder`); measures throughput across
batch-size × concurrency combinations and recommends settings for `features build`/training:

```bash
agritune encoder benchmark --base-url https://embeddings.precision.ai/v1 --model pai-embedding \
    --image-size 384 --batch-sizes 1 4 8 --concurrencies 1 2 4 --num-requests 5
```

Same command works unmodified for PhenoBench — the benchmark exercises the raw encoder API, not a
specific dataset's images (`--image-size` should still match your prepared size, `384` for both
datasets above).

### `features` — build, verify, inspect, clean, migrate

**Build** (resumable — re-running skips samples already cached):

```bash
agritune features build --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --base-url https://embeddings.precision.ai/v1 --model pai-embedding

agritune features build --manifest examples/datasets/phenobench/manifest.csv \
    --store examples/datasets/phenobench/features \
    --base-url https://embeddings.precision.ai/v1 --model pai-embedding
```

**Verify** integrity (corrupted/truncated shard detection):

```bash
agritune features verify --store examples/datasets/cwfid/features
agritune features verify --store examples/datasets/phenobench/features
```

**Inspect** (entry count, encoder models, patch dims present in the store):

```bash
agritune features inspect --store examples/datasets/cwfid/features
agritune features inspect --store examples/datasets/phenobench/features
```

**Clean** (remove stale/orphaned shard files no manifest entry references):

```bash
agritune features clean --store examples/datasets/cwfid/features
agritune features clean --store examples/datasets/phenobench/features
```

**Migrate** (e.g. `directory` → `sharded`, for moving from development to production scale):

```bash
agritune features migrate \
    --source examples/datasets/cwfid/features --source-type directory \
    --dest examples/datasets/cwfid/features-sharded --dest-type sharded --dest-entries-per-shard 500

agritune features migrate \
    --source examples/datasets/phenobench/features --source-type directory \
    --dest examples/datasets/phenobench/features-sharded --dest-type sharded --dest-entries-per-shard 500
```

A config's `feature_store_dir`/`store_type` must then point at the destination store, not the
source.

### `train`

```bash
agritune train --config examples/segmentation/cwfid.yaml

agritune train --config examples/segmentation/phenobench.yaml
```

Both configs already point at the `features`-built stores above and use the hosted encoder by
default — see the [Configuration reference](#configuration-reference) for every field they set,
and the [augmentation-mode](#tutorial-training-with-none--offline--online-augmentation) tutorial
below for switching `augmentation.mode`. Resuming an interrupted run is automatic: re-running the
same command with the same `run_id` picks up from `runs/<run_id>/checkpoints/last.ckpt` if one
exists.

### `evaluate`

```bash
agritune evaluate \
    --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --checkpoint examples/datasets/cwfid/runs/cwfid-sanity-mlp_probe-feature_aug/checkpoints/best.ckpt \
    --num-classes 3 --decoder mlp_probe \
    --encoder-model pai-embedding --encoder-revision "" \
    --batch-size 4

agritune evaluate \
    --manifest examples/datasets/phenobench/manifest.csv \
    --store examples/datasets/phenobench/features \
    --checkpoint examples/datasets/phenobench/runs/phenobench-sanity-token_fpn_1/checkpoints/best.ckpt \
    --num-classes 3 --decoder token_fpn --decoder-kwargs '{"cls_fusion": "film", "hidden_dim": 128, "num_layers": 2}' \
    --encoder-model pai-embedding --encoder-revision "" \
    --batch-size 32
```

`--decoder`/`--decoder-kwargs` must match the checkpoint's training run exactly (see its
`config.resolved.yaml`) or the state dict won't load. `--encoder-revision ""` matches how these
checkpoints were trained — the hosted `pai-embedding` API exposes no revision. Restrict to specific
samples with `--sample-ids`:

```bash
agritune evaluate --manifest examples/datasets/cwfid/manifest.csv --store examples/datasets/cwfid/features \
    --checkpoint examples/datasets/cwfid/runs/cwfid-sanity-mlp_probe-feature_aug/checkpoints/best.ckpt \
    --num-classes 3 --decoder mlp_probe --encoder-model pai-embedding --encoder-revision "" \
    --sample-ids cwfid-001 cwfid-002 cwfid-003
```

### `predict`

```bash
agritune predict \
    --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --checkpoint examples/datasets/cwfid/runs/cwfid-sanity-mlp_probe-feature_aug/checkpoints/best.ckpt \
    --num-classes 3 --decoder mlp_probe \
    --encoder-model pai-embedding --encoder-revision "" \
    --output examples/datasets/cwfid/predictions --overlays --overlay-alpha 0.5

agritune predict \
    --manifest examples/datasets/phenobench/manifest.csv \
    --store examples/datasets/phenobench/features \
    --checkpoint examples/datasets/phenobench/runs/phenobench-sanity-token_fpn_1/checkpoints/best.ckpt \
    --num-classes 3 --decoder token_fpn --decoder-kwargs '{"cls_fusion": "film", "hidden_dim": 128, "num_layers": 2}' \
    --encoder-model pai-embedding --encoder-revision "" \
    --output examples/datasets/phenobench/predictions --overlays --overlay-alpha 0.5
```

Writes `{sample_id}.png` (class map) and, with `--overlays`, `{sample_id}_overlay.png` per sample.
For a side-by-side viewer instead of raw PNGs:
[visualize_phenobench_predictions.py](segmentation/visualize_phenobench_predictions.py)
(`streamlit run examples/segmentation/visualize_phenobench_predictions.py`), which shows
prediction vs. ground truth for each already-predicted sample.

---

## Tutorial: training with `none` / `offline` / `online` augmentation

[examples/segmentation/cwfid.yaml](segmentation/cwfid.yaml) ships with `augmentation.mode: none`.
Override `augmentation.mode` (a dotlist key, since this is a flat, non-Hydra config) to try the
other modes — see [docs/augmentation.md](../docs/augmentation.md) for the full mechanics of each.

### `none` — no augmentation (the default)

Every sample passes through unchanged; `feature_provider: cached` is always valid since there is
only ever one cache key per sample.

```bash
agritune train --config examples/segmentation/cwfid.yaml run_id=cwfid-aug-none
```

### `offline` — one fixed augmented variant, reused every epoch

Deterministic per `(global_seed, sample_id, variant)` — the cache key incorporates that
fingerprint, so it stays valid across a whole run and composes with `feature_provider: cached` as
long as the store was already built for that exact augmentation config. Build it once with the
packaged offline augmentation config (its `geometric.resize: [798, 532]` is arbitrary — copy the
file and change it to match your own images, or drop `resize` entirely to keep each source image's
native size), then train against the cache with no further encoder calls:

```bash
agritune features build --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features-offline \
    --base-url https://embeddings.precision.ai/v1 --model pai-embedding \
    --augmentation-config precisionai/agritune/configs/augmentation/offline.yaml

agritune train --config examples/segmentation/cwfid.yaml \
    augmentation.mode=offline \
    feature_store_dir=examples/datasets/cwfid/features-offline \
    run_id=cwfid-aug-offline
```

Skip the separate `features build` step (and its `--augmentation-config`) and instead point
`feature_provider` at `hybrid` to populate the offline-variant cache lazily, on first use, from the
existing store:

```bash
agritune train --config examples/segmentation/cwfid.yaml \
    augmentation.mode=offline feature_provider=hybrid \
    run_id=cwfid-aug-offline-hybrid
```

### `online` — re-augmented fresh every epoch

Seeded from `(global_seed, epoch, sample_id, occurrence)`, so no two epochs see the same augmented
image — a `cached` provider's key would only ever match the first epoch's fingerprint, so this
mode **requires** `feature_provider: online` (or `hybrid`), which is validated up front and
rejected otherwise:

```bash
agritune train --config examples/segmentation/cwfid.yaml \
    augmentation.mode=online feature_provider=online \
    run_id=cwfid-aug-online
```

This calls the encoder on every batch of every epoch — expect it to run far slower than `none`/
`offline`, and to cost real encoder-API usage. `augmentation.mode=hybrid` (not shown here) is a
middle ground: it deterministically reuses an offline variant most of the time, and only takes the
fresh-online path for a configurable fraction of `(sample, epoch)` combinations.

The same three commands work unmodified against PhenoBench — swap
`examples/segmentation/cwfid.yaml` for `examples/segmentation/phenobench.yaml` and
`examples/datasets/cwfid/...` for `examples/datasets/phenobench/...` throughout.

---

## Tutorial: resize-only normalization vs. the full augmentation pipeline

These are two different code paths, not two settings of the same knob:

- **Resize-only** (dimensional normalization, not augmentation): both training *and* validation
  batches are *always* resized to `augmentation.geometric.resize` — regardless of
  `augmentation.mode`, including `none` — since `torch.stack` needs every mask in a batch to share
  one size, and this step never applies flips/crops/photometric transforms and never counts as
  "real" augmentation (it does not create an `AugmentationRecord`, so it never changes a cached
  feature's lookup key either). This is the *only* transform `mode: none` ever applies.
- **The full pipeline** (geometric + photometric transforms, `offline`/`online`/`hybrid` modes
  only): applied to *training* batches, starting with `geometric.resize` (if set, otherwise a
  no-op) and then layering random crop, flips, rotation, and photometric transforms on top — see
  [docs/augmentation.md](../docs/augmentation.md).

The consequence worth knowing: under `augmentation.mode: none` with `geometric.resize` left unset,
**training batches are not resized at all** — training relies on every image in the dataset already
sharing one native size (exactly what `prepare_cwfid.py --size 384`/`prepare_phenobench.py --size
384` guarantee). If your dataset's images are not already uniform in size, either set
`geometric.resize` (works under any mode, including `none`) or switch to `offline`/`online` for the
full pipeline, or pre-resize the dataset itself.

To see the difference directly: `examples/segmentation/cwfid.yaml`'s `augmentation.geometric.resize`
is `[256, 256]`, applied under `mode: none` to both training and validation batches alike. Switch to
`offline`/`online` and that same value becomes only the *first* step of every training sample's
augmentation — random crop, flips, rotation, and photometric transforms layer on top of it.
`examples/segmentation/phenobench.yaml` sets the same `[256, 256]`/`[224, 224]` resize/crop pair.

---

## Further reading

- [docs/augmentation.md](../docs/augmentation.md) — full mechanics of every augmentation mode and
  feature-space augmentation.
- [docs/configuration.md](../docs/configuration.md) — flat vs. Hydra config composition.
- [docs/feature-caching.md](../docs/feature-caching.md) — how the feature store keys augmented
  variants, and `DirectoryFeatureStore` vs. `ShardedFeatureStore`.
- [docs/reproducibility.md](../docs/reproducibility.md) — what `runs/<run_id>/` captures (resolved
  config, fingerprints, git/RNG state) and the `trainer.strict_resume` escape hatch for a confirmed
  benign checkpoint fingerprint mismatch.
- [SANITY_CHECK.md](SANITY_CHECK.md) — the single, linear walkthrough from raw dataset to
  prediction, including what a run directory contains.
- [FEATURE_TEST_COMMANDS.md](FEATURE_TEST_COMMANDS.md) — every decoder × provider × augmentation ×
  optimizer × scheduler combination, for exercising the full config surface at once.
