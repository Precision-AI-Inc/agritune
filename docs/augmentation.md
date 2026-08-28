# Augmentation

_Placeholder — expands as Phase 3 (augmentation subsystem) lands._

Image augmentation (`precisionai.agritune.augmentations.image`) composes
[albumentations](https://albumentations.ai/) transforms, separating geometric transforms (resize,
random crop, flips, rotation — applied to both image and mask, mask always with nearest
interpolation) from photometric and agricultural-domain-specific spectral transforms
(brightness/contrast/saturation/hue/blur/noise, sensor channel dropout, ground-sample-distance
jitter, a red/green vegetation-index-balance jitter, and a seasonal/growth-stage hue shift — image
only). Augmentation is deterministically seeded (`global_seed + sample_id + variant` offline,
`global_seed + epoch + sample_id + occurrence` online), applied via `Compose.set_random_seed`, so
the exact transformed image/mask can be reproduced from an `AugmentationRecord`. Modes: `none`,
`offline` (with `variants_per_sample`), `online`, and eventually `hybrid`. Feature-space
augmentation (patch dropout, token masking, Gaussian feature noise, CLS dropout, channel dropout —
`precisionai.agritune.augmentations.feature`) is independent and requires no extra encoder calls.
See `agritune_implementation_plan.md` §6, §21.
