# Datasets

_Placeholder — expands as Phase 2 (dataset system) lands._

Planned support: `ManifestDataset` / `SegmentationDataset` adapters over a CSV manifest
(`sample_id,image_path,mask_path,field_id,farm_id,capture_date`), with `random`, `grouped`
(`group_by: field_id`), and eventually `temporal` split strategies. Grouped splits matter for
agricultural data because nearby frames/images of the same field can be nearly identical — see
`agritune_implementation_plan.md` §5. `agritune dataset validate` / `agritune dataset inspect`
check for missing files, duplicate IDs, image/mask dimension mismatches, split leakage, and class
distribution.
