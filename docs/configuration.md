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
  - decoder: linear
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
fingerprint. `agritune_implementation_plan.md` §23's own example of an invalid combination.

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

## Group reference

| Group | Variants shipped | Selects |
|---|---|---|
| `dataset` | `agricultural_segmentation` | `manifest_path`, `num_classes` |
| `encoder` | `fake`, `remote` | `EncoderFingerprint` + `encoder_base_url`/`encoder_api_key` |
| `augmentation` | `none`, `offline`, `online`, `hybrid` | `AugmentationSelection` (mode, geometric/photometric transforms) |
| `feature_provider` | `cached`, `online`, `hybrid` | which `FeatureProvider` `run_training` builds |
| `task` | `segmentation` | loss config, `val_metric_name`, `higher_is_better` |
| `decoder` | `linear`, `token_fpn` | `decoder_name` |
| `optimizer` | `adamw`, `adam`, `sgd` | `OptimizerConfig` |
| `scheduler` | `none`, `cosine`, `cosine_warmup`, `linear_warmup`, `polynomial`, `plateau` | `SchedulerConfig`, or no scheduler |
| `tracking` | `null`, `jsonl`, `tensorboard`, `local` (jsonl+tensorboard), `mlflow`, `wandb`, `neptune`, `comet` | `TrackingSelection` |

Every group's YAML lives in `precisionai/agritune/configs/<group>/<variant>.yaml` — copy one to
add a new named preset (e.g. a second `dataset/` variant for a different farm's manifest).
