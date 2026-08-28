# Augmentation

_Placeholder — expands as Phase 3 (augmentation subsystem) lands._

Image augmentation (`precisionai.agritune.augmentations.image`) separates geometric transforms
(resize, random crop, flips, rotation — applied to both image and mask, mask always with nearest
interpolation) from photometric transforms (brightness/contrast/saturation/hue/blur/noise — image
only). Augmentation is deterministically seeded (`global_seed + sample_id + variant` offline,
`global_seed + epoch + sample_id + occurrence` online) so the exact transformed image/mask can be
reproduced from an `AugmentationRecord`. Modes: `none`, `offline` (with `variants_per_sample`),
`online`, and eventually `hybrid`. Feature-space augmentation (patch dropout, token masking,
Gaussian feature noise, CLS dropout, channel dropout — `precisionai.agritune.augmentations.feature`)
is independent and requires no extra encoder calls. See `agritune_implementation_plan.md` §6, §21.
