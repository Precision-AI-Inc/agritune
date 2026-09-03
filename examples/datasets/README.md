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

Pass `--full` instead of `--max-samples` to download all 60 frames:

```bash
python examples/datasets/prepare_cwfid.py --output examples/datasets/cwfid --full --size 384
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

## Worked example: PhenoBench (bigger, real-world sugar-beet fields)

[PhenoBench](https://www.phenobench.org) (Weyler et al., IEEE TPAMI 2024) is a much larger
real-world counterpart to CWFID: 1,407 training and 772 validation UAV images (1024×1024) of
sugar-beet fields across multiple growth stages, with the same crop/weed segmentation task. Use it
to sanity-check a model at a scale closer to production before committing to a full training run.

Unlike CWFID, PhenoBench ships as a single ~7.6 GB archive rather than per-file downloads, so the
prep script downloads it once (cached, so a later re-run with different `--max-samples`/`--size`
does not re-download) and converts a chosen number of samples per split:

```bash
python examples/datasets/prepare_phenobench.py \
    --output examples/datasets/phenobench \
    --max-samples 40 \
    --size 384
```

Pass `--full` to convert every labeled image (1,407 train + 772 val = 2,179 samples) instead of the
default 40-per-split subset — the archive download is the same size either way, `--full` only
changes how many of the downloaded images get converted:

```bash
python examples/datasets/prepare_phenobench.py --output examples/datasets/phenobench --full --size 384
```

That produces the same layout as CWFID:

```text
examples/datasets/phenobench/
├── .cache/PhenoBench-v110.zip   # cached source archive, reused across runs
├── images/train_05-15_00028_P0030852.png …
├── masks/train_05-15_00028_P0030852.png  …
└── manifest.csv
```

Class map (PhenoBench's own partial-visibility labels 3/4 are collapsed into crop/weed so the
manifest matches CWFID's convention exactly):

| Index | Class |
|---|---|
| 0 | background |
| 1 | crop |
| 2 | weed |

PhenoBench's `test` split annotations are withheld by the authors for their leaderboard, so this
script only converts `train`/`val` (`--splits train val` by default).

PhenoBench is licensed CC BY-SA 4.0 — derivatives (including the files this script writes) carry
the same license. If you use it, cite:

> Weyler, J., Magistri, F., Marks, E., Chong, Y.L., Sodano, M., Roggiolani, G., Chebrolu, N.,
> Stachniss, C. and Behley, J. (2024). PhenoBench: A Large Dataset and Benchmarks for Semantic
> Image Interpretation in the Agricultural Domain. *IEEE Transactions on Pattern Analysis and
> Machine Intelligence*.

The prepared tree (including the cached archive) is gitignored.

## Adapting this to your own data

1. Put RGB images and integer class masks in two folders.
2. Resize so each image/mask pair shares H×W (AgriTune's validator flags mismatches).
3. Write a CSV with at least `sample_id,image_path,mask_path`.
4. `agritune dataset validate --manifest ... --num-classes N` then `agritune dataset inspect`.

The full command sequence after this step is in [../SANITY_CHECK.md](../SANITY_CHECK.md).
