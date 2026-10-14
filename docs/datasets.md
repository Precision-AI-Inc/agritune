# Datasets

`ManifestDataset` (satisfying the `SegmentationDataset` protocol) adapts a CSV manifest
(`sample_id,image_path,mask_path,field_id,farm_id,capture_date,...` — any extra column becomes
per-sample metadata) into `Sample` objects. Three split strategies (`precisionai.agritune.data.split`):

- `random_split` — assigns each sample independently, by a seeded hash of its sample ID.
- `grouped_split` — keeps every sample sharing a metadata key (e.g. `field_id`) in the same split.
  Matters for agricultural data because nearby frames/images of the same field can be nearly
  identical — a random split would leak near-duplicates across train/val/test.
- `temporal_split` — orders samples by a date metadata field and assigns the earliest to train,
  latest to test; not seeded, since order is fully determined by the date field.

`detect_group_leakage` flags group values that ended up split across more than one subset, useful
even on a split produced by `random_split`. `agritune dataset validate` / `agritune dataset
inspect` check for missing files, duplicate IDs, image/mask dimension mismatches, invalid labels,
and report class-pixel-count statistics.

`agritune dataset init --output manifest.csv` writes an example manifest with a header row and a
few placeholder samples (`sample_id`, `image_path`, `mask_path`, plus `field_id`/`farm_id`/
`capture_date` metadata) to start from — replace the placeholder rows with your own samples, then
point `--manifest`/`manifest_path` at the result. Pass `--force` to overwrite an existing file.

## Train, validation, and test manifests

`agritune train` splits its own manifest. Every row of `manifest_path` is assigned to either
training or validation by `random_split`, a seeded hash of each sample ID: `val_fraction` of the
samples (default `0.2`) go to validation and the rest to training. Any `split` column in the
manifest is ignored, so rows marked `test` are trained on like any other row.

Keep test samples in a **separate manifest** that the training manifest does not list, and
evaluate on it with `agritune evaluate --manifest`:

| Manifest | Used by | Contents |
|---|---|---|
| training manifest (`manifest_path`) | `agritune train` | Every sample for training *and* validation; the run splits it by `val_fraction` |
| test manifest | `agritune evaluate`, `agritune predict` | Held-out samples only; never listed in the training manifest |

Both manifests can share one feature store: cache keys depend on the sample, its image, and the
encoder, not on which manifest listed it. Run `agritune features build` once per manifest against
the same `--store`.

The training/validation split is per sample, not per group. Tiles cropped from the same image, or
near-identical frames of the same field, can land on both sides, which makes validation metrics
optimistic. Hold out whole groups in the test manifest to get a score that reflects unseen
fields. For example, with `grouped_split` on a `field_id` column:

```python
import csv

from precisionai.agritune.data.manifest import load_manifest
from precisionai.agritune.data.split import grouped_split

split = grouped_split(load_manifest("manifest_all.csv"), group_by="field_id", train_fraction=0.9, val_fraction=0.0)
test_ids = set(split.test)

with open("manifest_all.csv", newline="") as source:
    reader = csv.DictReader(source)
    rows, fields = list(reader), reader.fieldnames

# Written next to manifest_all.csv, so every relative image/mask path still resolves.
for name, keep in (("manifest_trainval.csv", False), ("manifest_test.csv", True)):
    with open(name, "w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if (row["sample_id"] in test_ids) is keep)
```

Train on `manifest_trainval.csv` and evaluate on `manifest_test.csv`. Each run records the exact
train/validation assignment it used in `runs/<run_id>/dataset.json`.

## Preparing a dataset

A worked example that downloads the public Crop/Weed Field Image Dataset (CWFID), converts its
RGB annotations into single-channel class-index masks, and writes a manifest lives in
[`examples/datasets/`](../examples/datasets/README.md). The end-to-end command sequence (validate
→ feature build → train → evaluate → predict) is [`examples/SANITY_CHECK.md`](../examples/SANITY_CHECK.md).
