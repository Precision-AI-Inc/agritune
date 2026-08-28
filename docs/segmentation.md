# Segmentation

_Placeholder — expands as Phase 8 (segmentation task) lands._

The linear-probe decoder (patch tokens → linear projection → per-patch class logits → reshape via
`patch_grid` → upsample) is the diagnostic baseline: if a more complex decoder doesn't beat it,
something upstream is wrong. TokenFPN (channel projection → spatial reshape → pseudo feature
pyramid → convolutional refinement → upsampling) is the stronger decoder, with optional CLS fusion
(`none` / `concat` / `film`) — CLS is never required. Losses: CrossEntropy, BCEWithLogits, Dice, and
CE/BCE + Dice combinations, with `ignore_index` and class-weight support. Metrics: mean IoU,
per-class IoU, Dice/F1, pixel accuracy, confusion matrix. See `agritune_implementation_plan.md` §11.
