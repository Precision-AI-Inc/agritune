# Segmentation

Six decoders are available (selected via `decoder_name` / the `decoder` Hydra config group), all
consuming the same frozen-encoder `EncoderFeatures` and producing the same per-pixel class-logit
tensor `(B, num_classes, H, W)`:

- **`mlp_probe`** (`MLPProbeDecoder`) — patch tokens → a stack of per-patch linear (+ ReLU) layers
  → per-patch class logits → reshape via `patch_grid` → upsample. The diagnostic baseline: with
  `hidden_dims=()` (the default) it is a single linear projection, and if a more complex decoder
  doesn't beat it, something upstream is wrong. `hidden_dims=(256,)` (etc.) adds non-linear
  capacity without changing the decoder's identity.
- **`token_fpn`** (`TokenFPNDecoder`) — channel projection → spatial reshape → pseudo feature
  pyramid → convolutional refinement → upsampling, with optional CLS fusion (`cls_fusion`:
  `none` / `concat` / `film`) — CLS is never required.
- **`aspp`** (`ASPPDecoder`) — Atrous Spatial Pyramid Pooling: parallel dilated convolutions at
  several rates (`atrous_rates`) plus a global-context branch, fused into per-pixel logits.
  Multi-scale context from the single patch-grid resolution the frozen features expose.
- **`ppm`** (`PyramidPoolingDecoder`) — PSPNet-style pyramid pooling: parallel adaptive-average-pool
  bins (`pool_sizes`), upsampled and concatenated with the original grid, then convolutionally
  fused into per-pixel logits.
- **`segmenter`** (`SegmenterMaskTransformerDecoder`) — Segmenter-style mask transformer: learnable
  per-class embeddings jointly processed with the patch tokens through shared self-attention
  transformer layers, then a scaled dot product between the two yields per-pixel mask logits
  directly (no convolutional head).
- **`mask_former`** (`MaskFormerDecoder`) — MaskFormer/Mask2Former-style mask classification: a
  fixed set of learnable object queries (`num_queries`, independent of `num_classes`) cross-attends
  to the patch tokens through transformer decoder layers; each query's class distribution and mask
  embedding combine into dense per-pixel class scores. Trains against the same per-pixel loss as
  every other decoder here, rather than the original papers' bipartite-matching set loss.

Losses: CrossEntropy, BCEWithLogits, Dice, and CE/BCE + Dice combinations, with `ignore_index` and
class-weight support. Metrics (`SegmentationMetric`): mean/per-class IoU, precision, recall,
Dice/F1, pixel accuracy, and a confusion matrix.

## Visualization

`precisionai.agritune.tasks.segmentation.visualization` colorizes a per-pixel class-index map
(`default_palette`/`colorize_predictions`, one distinct color per class around the HSV hue wheel)
and alpha-blends it over the original image (`overlay_predictions_on_image`, nearest-neighbor
resized to the image's resolution so it aligns without smearing class boundaries); `side_by_side`
composes panels for quick comparison. `agritune predict --overlays` writes a
`{sample_id}_overlay.png` alongside each raw prediction PNG.
