# Augmentation

Image augmentation (`precisionai.agritune.augmentations.image`) composes
[albumentations](https://albumentations.ai/) transforms, separating geometric transforms (resize,
random crop, flips, rotation — applied to both image and mask, mask always with nearest
interpolation) from photometric and agricultural-domain-specific spectral transforms
(brightness/contrast/saturation/hue/blur/noise, sensor channel dropout, ground-sample-distance
jitter, a red/green vegetation-index-balance jitter, and a seasonal/growth-stage hue shift — image
only). Augmentation is deterministically seeded, applied via `Compose.set_random_seed`, so the
exact transformed image/mask can be reproduced from an `AugmentationRecord`. Modes:

- `none` — passes the sample through unchanged.
- `offline` — a fixed pool of `variants_per_sample` variants, seeded from `global_seed + sample_id
  + variant`; compatible with an offline feature cache.
- `online` — re-augmented every epoch, seeded from `global_seed + epoch + sample_id + occurrence`.
- `hybrid` — deterministically reuses an offline variant most of the time, and derives a fresh
  online seed for a configurable fraction (`hybrid_online_probability`) of `(sample, epoch,
  occurrence)` combinations — see `derive_hybrid_seed`.

Feature-space augmentation (patch dropout, token masking, Gaussian feature noise, CLS dropout,
channel dropout — `precisionai.agritune.augmentations.feature`) is independent, requires no extra
encoder calls, and is applied by `Trainer` as an optional train-only step (never during
validation).
