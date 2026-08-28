# Configuration

_Placeholder — expands as Phase 0/23 (config system) lands._

AgriTune composes configuration with Hydra/OmegaConf under `precisionai/agritune/configs/`:
`config.yaml` plus config groups (`dataset/`, `encoder/`, `augmentation/`, `feature_provider/`,
`task/`, `decoder/`, `optimizer/`, `scheduler/`, `tracking/`). Example composition:

```yaml
defaults:
  - dataset: agricultural_segmentation
  - encoder: remote
  - augmentation: offline
  - feature_provider: cached
  - task: segmentation
  - decoder: token_fpn
  - optimizer: adamw
  - scheduler: cosine
  - tracking: local
```

Overridden from the CLI the same way Hydra always allows:

```bash
agritune train augmentation=online feature_provider=online decoder=linear
```

Invalid combinations (e.g. `augmentation=online` with `feature_provider=cached`) are rejected at
config-validation time, before training starts — see `agritune_implementation_plan.md` §23.
