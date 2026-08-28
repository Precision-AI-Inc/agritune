# Segmentation

The linear-probe decoder (patch tokens → linear projection → per-patch class logits → reshape via
`patch_grid` → upsample) is the diagnostic baseline: if a more complex decoder doesn't beat it,
something upstream is wrong. TokenFPN (channel projection → spatial reshape → pseudo feature
pyramid → convolutional refinement → upsampling) is the stronger decoder, with optional CLS fusion
(`none` / `concat` / `film`) — CLS is never required. Losses: CrossEntropy, BCEWithLogits, Dice, and
CE/BCE + Dice combinations, with `ignore_index` and class-weight support. Metrics
(`SegmentationMetric`): mean/per-class IoU, precision, recall, Dice/F1, pixel accuracy, and a
confusion matrix. See `agritune_implementation_plan.md` §11.

## Visualization

`precisionai.agritune.tasks.segmentation.visualization` colorizes a per-pixel class-index map
(`default_palette`/`colorize_predictions`, one distinct color per class around the HSV hue wheel)
and alpha-blends it over the original image (`overlay_predictions_on_image`, nearest-neighbor
resized to the image's resolution so it aligns without smearing class boundaries); `side_by_side`
composes panels for quick comparison. `agritune predict --overlays` writes a
`{sample_id}_overlay.png` alongside each raw prediction PNG.
