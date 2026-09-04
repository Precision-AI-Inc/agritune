# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Pull-request CI now requests `contents: read` only, reusable unit-test/pre-commit jobs declare
  the same, and their checkouts set `persist-credentials: false` so pull-request code cannot run
  with a repository write token.
- Logging redacts API keys, Authorization values, sensitive Hydra/dotlist overrides, and URL
  userinfo/query credentials before emit, including DEBUG override dumps and encoder base URLs.

### Fixed

- Pyright optional-palette subscript in `colorize_predictions` and live-test API-key narrowing
  after `pytest.skip`.

### Added

- Info/debug/warning logging across previously-silent modules (services, encoder, evaluator),
  plus `tqdm` progress bars over every long-running loop (training/validation batches,
  evaluation, prediction, feature precompute, encoder benchmarking, and the CWFID/PhenoBench prep
  scripts) via a new `precisionai.agritune.logging.progress_iter` helper. Progress bars are
  opt-in per call (`show_progress`, default `False`) so the API stays headless; `Trainer`/
  `evaluate()` gate them on `DistributedContext.is_main_process`, ready for the eventual DDP
  rollout. The FastAPI app now calls `configure_logging()` on startup
  (`AGRITUNE_LOG_LEVEL`, default `INFO`), and its Rich console handler now writes to stderr
  instead of stdout so log lines never interleave with a CLI command's own JSON output.
- CWFID example: `examples/datasets/prepare_cwfid.py` downloads the public Crop/Weed Field Image
  Dataset, writes an AgriTune manifest, and `examples/SANITY_CHECK.md` runs the full hosted-encoder
  pipeline against it. A `--full` flag downloads all 60 frames without needing to know the exact
  count.
- PhenoBench example: `examples/datasets/prepare_phenobench.py` downloads the public PhenoBench
  sugar-beet dataset (1,407 train / 772 val real UAV field images, vs. CWFID's 60) and writes an
  AgriTune manifest, collapsing its 5-way partial-visibility labels into AgriTune's 3-class
  background/crop/weed convention.
- Optional `.env` support (`pip install pai-agritune[dotenv]`): a `.env` file is loaded into the
  environment before Hydra composition, so `${oc.env:AGRITUNE_ENCODER_API_KEY,null}` in
  `encoder/remote.yaml` resolves from it. Exported shell variables still take precedence, and
  `AGRITUNE_ENV_FILE` selects a file outside the working directory.
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
- `Trainer.fit`'s optional `train_metric` parameter: a `Metric` accumulated over every training
  batch each epoch (reset at the epoch's start) and logged as `train_{name}` (e.g. `train_mean_iou`)
  alongside the existing `train_loss`/`lr`, exposed via `Trainer.last_train_metrics`. `run_training`
  now always passes one, so `TrainingRunResult.train_metrics` / `agritune train`'s printed summary
  / `POST /train`'s response report full segmentation metrics for the training set too, not just
  loss — useful for spotting a train/val gap, though noisier than validation since it reflects a
  moving model on (possibly augmented) training batches.
- `Evaluator.evaluate` (shared by `Trainer`'s validation pass and `agritune evaluate`) now also
  computes a batch-size-weighted mean loss and adds it to its returned dict as `"loss"`, so
  `val_metric_name: loss` (with `higher_is_better: false`) works as a checkpoint/early-stopping
  signal. `EvaluationRunConfig`/`agritune evaluate --loss-*`/`POST /evaluate`'s `loss` field let the
  loss config used for this be set to match what a checkpoint was actually trained under.
- `agritune features build --augmentation-config <path> --seed <int>` (and `POST /features/build`'s
  matching `augmentation_config_path`/`seed` fields): precomputes a deterministic offline-augmented
  variant alongside the unaugmented features, closing a gap where `feature_service.build_features`
  already supported `augmentation_mode=OFFLINE` but neither entry point exposed it. The
  `--augmentation-config` file is the same `mode`/`variant`/`geometric`/`photometric` shape as
  `configs/augmentation/*.yaml` — point both the build and the training config's `augmentation:`
  block at the same file so their cache keys match. New `cli.config.load_augmentation_selection`
  shares the parsing between the CLI and the API route.
- `TrainingRunConfig.feature_augmentation` (`feature_augmentation:` in a flat YAML config, or the
  new `feature_augmentation` Hydra group — `none`/`light` — in the composed one), closing a gap
  where `FeatureAugmentationPipeline`/`FeatureAugmentationConfig` (patch dropout, token masking,
  Gaussian feature noise, CLS dropout, channel dropout) were fully implemented and accepted by
  `Trainer`'s constructor but never reachable from `agritune train`/`POST /train` — only from
  direct Python construction. Included in the run's critical-config reproducibility fingerprint,
  alongside image-space `augmentation`.
- `agritune config init --output <path> [--force]`: writes a fully-commented, flat training config
  template spelling out every field `TrainingRunConfig` accepts at its default value, with the
  dataset-specific fields marked `REQUIRED` as placeholders — the same shape as
  `examples/segmentation/cwfid.yaml` without the CWFID-specific values. Backed by the new
  `services.config_template_service`.
- `agritune dataset init --output <path> [--force]`: writes an example manifest CSV (a header row
  plus a few placeholder samples) to start a new dataset's manifest from. Backed by
  `services.dataset_service.write_manifest_template`, sharing its file-writing/overwrite-guard
  logic with `config init` via the new `utils.scaffold` helper.

### Changed

- The image augmentation pipeline (`augmentations.image`) now composes
  [albumentations](https://albumentations.ai/) transforms instead of hand-rolled PIL/numpy code,
  using `Compose.set_random_seed` for the same exact-reproducibility guarantee as before.

### Fixed

- Flush partial gradient-accumulation windows instead of silently dropping their gradients.
- Execute gateway-split encoder batches concurrently while preserving response order.
- Preserve augmentation fingerprints through training batches so hybrid cache keys cannot reuse
  stale features across different augmented images.
- Harden variable-grid feature masking, remote-response validation, checkpoint fingerprints,
  secret-safe run provenance, and tracker-failure isolation.
- Persist early-stopping progress across resume so training cannot run extra epochs after a stop.
- Derive feature-cache keys from a canonical JSON payload so delimiter characters in sample IDs
  cannot collide with adjacent fingerprint fields.
- Decode CLS embeddings from little-endian float32 base64, matching `encoding_format="base64"` on
  the hosted encoder API — a float-list-only parser crashed on the real `pai-embedding` response.
- Defer `comet_ml`'s import to `CometTracker` construction instead of importing it at module load.
  `comet_ml` auto-instruments other frameworks in the process (notably `mlflow`) merely by being
  imported; since `tracking_selection.py` imports every tracker module up front, selecting any
  *other* backend (e.g. `mlflow`) silently created an uninvited offline Comet experiment whenever
  `comet_ml` happened to be installed alongside it — which it always is, since both ship in the
  `pai-agritune[tracking]` extra.
- Derive a training run's decoder `output_size` from the actually-augmented probe sample instead
  of the raw pre-augmentation one, so `augmentation.geometric.resize`/`random_crop` (which change
  spatial size) no longer builds a decoder upsampling to the wrong resolution.
  `SegmentationLoss.forward`/`SegmentationMetric.update` now also bilinearly resize logits to the
  target's spatial size when they differ, since validation targets are never augmented and can
  therefore legitimately disagree with training's working resolution under one fixed decoder.
- Log every encoder retry attempt (`EncoderGateway`) at `WARNING` instead of only counting it in
  metrics — a bounded retry-then-fail run and an actual hang were previously indistinguishable from
  the terminal, since nothing was printed during the backoff sleeps between attempts.
- CI (`unit-test.yml`/`pre-commit.yml`): install the CPU-only `torch` wheel
  (`--index-url https://download.pytorch.org/whl/cpu`) before `pip install -e ".[dev]"` — the
  default Linux PyPI wheel bundles the full CUDA runtime (several GB of `nvidia-*`/`triton`
  packages) that a CPU-only CI runner never uses, and was exhausting the runner's disk.
- `evaluation_service.py`/`prediction_service.py`/`training_service.py`: build a decoder's
  `output_size` via a new `segmentation_common.mask_output_size` helper that returns a concrete
  `tuple[int, int]`, instead of `tuple(np.array(mask).shape)` (typed as a variable-length
  `tuple[int, ...]`) — fixes a `pyright` `reportArgumentType` error that a newer `numpy` type-stub
  resolution surfaces on some Python versions.
- `trailing-whitespace` pre-commit hook: pass `--markdown-linebreak-ext=md` so it stops stripping
  intentional Markdown hard-break trailing spaces (it was rewriting `examples/SANITY_CHECK.md` on
  every run).
- Fix the remaining pre-existing `pyright` errors, none introduced by recent work but all blocking
  a clean `pre-commit`/CI run: `LossConfigRequest.name` (API schema) now uses the same `LossName`
  literal as `SegmentationLossConfig` instead of a bare `str`; `ImageAugmentationPipeline` builds
  its transform list typed as albumentations' own `TransformsSeqType` instead of the invariant-
  incompatible `list[BasicTransform]`; `PrecisionContext.autocast()` is typed to yield `Any`
  (`nullcontext`/`torch.autocast` disagree on what `__enter__` returns, but no caller binds it);
  `cli/config.py`'s `_config_from_dict(resolved)` call gets the same `# type: ignore[arg-type]`
  treatment the file already uses for `OmegaConf.to_container`'s deliberately-broad return type;
  and several tests narrow an `Optional`/generic-`nn.Module` value with an `assert ... is not None`
  / `isinstance` check before using it, instead of relying on runtime knowledge pyright can't see.
- Untrack `.claude/scheduled_tasks.lock` (a Claude Code runtime lock file — session ID/PID/
  timestamp, changes every session) and gitignore it; it was flapping `end-of-file-fixer` on
  unrelated commits.

[Unreleased]: https://github.com/Precision-AI-Inc/agritune/compare/v0.1.0...HEAD
