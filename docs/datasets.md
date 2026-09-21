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

## Preparing a dataset

A worked example that downloads the public Crop/Weed Field Image Dataset (CWFID), converts its
RGB annotations into single-channel class-index masks, and writes a manifest lives in
[`examples/datasets/`](../examples/datasets/README.md). The end-to-end command sequence (validate
→ feature build → train → evaluate → predict) is [`examples/SANITY_CHECK.md`](../examples/SANITY_CHECK.md).
