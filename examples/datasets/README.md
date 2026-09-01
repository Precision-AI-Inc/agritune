# Preparing a segmentation dataset for AgriTune

AgriTune does not read vendor-specific dataset folders. It reads a **CSV manifest** whose paths
are relative to the manifest file itself:

```text
sample_id,image_path,mask_path,field_id,farm_id,capture_date,...
cwfid-001,images/001.png,masks/001.png,sugar-beet,...
```

Required columns:

| Column | Meaning |
|---|---|
| `sample_id` | Stable unique id (used in cache keys, predictions, provenance) |
| `image_path` | RGB image, relative to the manifest directory |
| `mask_path` | Single-channel class-index PNG, same H×W as the image |

Every extra column becomes per-sample metadata (`field_id` is what grouped splits use). Mask
pixels are integer class ids in `[0, num_classes)`. See `docs/datasets.md`.

## Worked example: CWFID (crop / weed)

[CWFID](https://github.com/cwfid/dataset) is a 60-image sugar-beet field set with colour-coded
crop/weed maps (Haug & Ostermann, ECCV 2014 Workshops). The images are **not** in this repository
— the prep script downloads them and writes AgriTune-ready files:

```bash
python examples/datasets/prepare_cwfid.py \
    --output examples/datasets/cwfid \
    --max-samples 24 \
    --size 384
```

That produces:

```text
examples/datasets/cwfid/
├── images/001.png …          # RGB, resized square
├── masks/001.png  …          # class index 0=background, 1=crop, 2=weed
└── manifest.csv
```

Class map:

| Index | Class |
|---|---|
| 0 | background |
| 1 | crop |
| 2 | weed |

CWFID is research-use / citation-required. If you use it, cite:

> Haug, S. and Ostermann, J. (2015). A Crop/Weed Field Image Dataset for the Evaluation of
> Computer Vision Based Precision Agriculture Tasks. *ECCV 2014 Workshops*.

The prepared tree is gitignored. Re-run the script on a new machine.

## Adapting this to your own data

1. Put RGB images and integer class masks in two folders.
2. Resize so each image/mask pair shares H×W (AgriTune's validator flags mismatches).
3. Write a CSV with at least `sample_id,image_path,mask_path`.
4. `agritune dataset validate --manifest ... --num-classes N` then `agritune dataset inspect`.

The full command sequence after this step is in [../SANITY_CHECK.md](../SANITY_CHECK.md).
