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
and report class-pixel-count statistics. See `agritune_implementation_plan.md` §5.
