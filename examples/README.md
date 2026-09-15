# AgriTune tutorials

Task-oriented walkthroughs for the packaged examples, built on the public
[CWFID](https://github.com/cwfid/dataset) dataset prepared in the root [README](../README.md)'s
Quick Start. Every command below assumes that's already done:

```bash
python examples/datasets/prepare_cwfid.py --output examples/datasets/cwfid --max-samples 24 --size 384
agritune features build --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --base-url https://embeddings.precision.ai/v1 --model pai-embedding
```

No encoder access? Swap `--base-url ... --model pai-embedding` for `--model fake-encoder` anywhere
below — every command still runs, against synthetic features.

See also: [datasets/README.md](datasets/README.md) (preparing CWFID/PhenoBench from scratch),
[SANITY_CHECK.md](SANITY_CHECK.md) (the full single-path walkthrough), and
[FEATURE_TEST_COMMANDS.md](FEATURE_TEST_COMMANDS.md) (an exhaustive command matrix across every
decoder/provider/optimizer/scheduler combination).

---

## Tutorial: training with `none` / `offline` / `online` augmentation

[examples/segmentation/cwfid.yaml](segmentation/cwfid.yaml) ships with `augmentation.mode: none`.
Override `augmentation.mode` (a dotlist key, since this is a flat, non-Hydra config) to try the
other modes — see [docs/augmentation.md](../docs/augmentation.md) for the full mechanics of each.

### `none` — no augmentation (the default)

Every sample passes through unchanged; `feature_provider: cached` is always valid since there is
only ever one cache key per sample.

```bash
agritune train --config examples/segmentation/cwfid.yaml run_id=cwfid-aug-none
```

### `offline` — one fixed augmented variant, reused every epoch

Deterministic per `(global_seed, sample_id, variant)` — the cache key incorporates that
fingerprint, so it stays valid across a whole run and composes with `feature_provider: cached` as
long as the store was already built for that exact augmentation config. Build it once with the
packaged offline augmentation config (its `geometric.resize: [798, 532]` is arbitrary — copy the
file and change it to match your own images, or drop `resize` entirely to keep each source image's
native size), then train against the cache with no further encoder calls:

```bash
agritune features build --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features-offline \
    --base-url https://embeddings.precision.ai/v1 --model pai-embedding \
    --augmentation-config precisionai/agritune/configs/augmentation/offline.yaml

agritune train --config examples/segmentation/cwfid.yaml \
    augmentation.mode=offline \
    feature_store_dir=examples/datasets/cwfid/features-offline \
    run_id=cwfid-aug-offline
```

Skip the separate `features build` step (and its `--augmentation-config`) and instead point
`feature_provider` at `hybrid` to populate the offline-variant cache lazily, on first use, from the
existing store:

```bash
agritune train --config examples/segmentation/cwfid.yaml \
    augmentation.mode=offline feature_provider=hybrid \
    run_id=cwfid-aug-offline-hybrid
```

### `online` — re-augmented fresh every epoch

Seeded from `(global_seed, epoch, sample_id, occurrence)`, so no two epochs see the same augmented
image — a `cached` provider's key would only ever match the first epoch's fingerprint, so this
mode **requires** `feature_provider: online` (or `hybrid`), which is validated up front and
rejected otherwise:

```bash
agritune train --config examples/segmentation/cwfid.yaml \
    augmentation.mode=online feature_provider=online \
    run_id=cwfid-aug-online
```

This calls the encoder on every batch of every epoch — expect it to run far slower than `none`/
`offline`, and to cost real encoder-API usage. `augmentation.mode=hybrid` (not shown here) is a
middle ground: it deterministically reuses an offline variant most of the time, and only takes the
fresh-online path for a configurable fraction of `(sample, epoch)` combinations.

---

## Tutorial: resize-only normalization vs. the full augmentation pipeline

These are two different code paths, not two settings of the same knob:

- **Resize-only** (dimensional normalization, not augmentation): validation batches are *always*
  resized to `augmentation.geometric.resize` — regardless of `augmentation.mode` — since
  `torch.stack` needs every mask in a batch to share one size, and validation must never apply
  flips/crops/photometric transforms. This is the only resizing that ever touches validation data.
- **The full pipeline** (geometric + photometric transforms, `offline`/`online`/`hybrid` modes
  only): applied to *training* batches, starting with `geometric.resize` (if set) and then
  layering random crop, flips, rotation, and photometric transforms on top — see
  [docs/augmentation.md](../docs/augmentation.md).

The consequence worth knowing: under `augmentation.mode: none`, **training batches are not resized
at all** — training relies on every image in the dataset already sharing one native size (exactly
what `prepare_cwfid.py --size 384`/`prepare_phenobench.py --size 384` guarantee). If your dataset's
images are not already uniform in size, `none` will fail to `torch.stack` a training batch — switch
to `offline`/`online` (whose pipeline resizes training images too) or pre-resize the dataset
itself.

To see the difference directly: `examples/segmentation/cwfid.yaml`'s `augmentation.geometric.resize`
is `[256, 256]`, but since `mode: none` "ignores every field below it" for training, that value is
only ever consulted for validation. Switch to `offline`/`online` (as above) and it also governs the
first step of every training sample's augmentation.

---

## Tutorial: evaluation

`agritune evaluate` runs the same feature-provider/decoder stack as training, with no gradient
updates, against a saved checkpoint:

```bash
agritune evaluate \
    --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --checkpoint examples/datasets/cwfid/runs/cwfid-aug-none/checkpoints/best.ckpt \
    --num-classes 3 --decoder mlp_probe \
    --encoder-model pai-embedding --encoder-revision "" \
    --batch-size 4
```

Restrict to specific samples (e.g. a held-out debugging subset) with `--sample-ids`:

```bash
agritune evaluate \
    --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --checkpoint examples/datasets/cwfid/runs/cwfid-aug-none/checkpoints/best.ckpt \
    --num-classes 3 --decoder mlp_probe \
    --encoder-model pai-embedding --encoder-revision "" \
    --sample-ids cwfid-001 cwfid-002 cwfid-003
```

`--encoder-revision ""` matches how these checkpoints were trained — the hosted `pai-embedding` API
exposes no revision, so the cache/checkpoint fingerprint was built with `revision=None`.

---

## Tutorial: prediction (class maps + overlays)

`agritune predict` writes one class-index PNG per sample (plus, with `--overlays`, a colorized
overlay blended onto the source image) instead of computing metrics:

```bash
agritune predict \
    --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --checkpoint examples/datasets/cwfid/runs/cwfid-aug-none/checkpoints/best.ckpt \
    --num-classes 3 --decoder mlp_probe \
    --encoder-model pai-embedding --encoder-revision "" \
    --output examples/datasets/cwfid/predictions \
    --overlays --overlay-alpha 0.5
```

This writes `examples/datasets/cwfid/predictions/{sample_id}.png` (class map) and
`{sample_id}_overlay.png` (overlay). For a side-by-side viewer instead of raw PNGs, see
[visualize_phenobench_predictions.py](segmentation/visualize_phenobench_predictions.py) — a
Streamlit app (`streamlit run examples/segmentation/visualize_phenobench_predictions.py`) that
shows prediction vs. ground truth for each PhenoBench sample already predicted to disk.

---

## Further reading

- [docs/augmentation.md](../docs/augmentation.md) — full mechanics of every augmentation mode and
  feature-space augmentation.
- [docs/configuration.md](../docs/configuration.md) — flat vs. Hydra config composition.
- [docs/feature-caching.md](../docs/feature-caching.md) — how the feature store keys augmented
  variants, and `DirectoryFeatureStore` vs. `ShardedFeatureStore`.
- [SANITY_CHECK.md](SANITY_CHECK.md) — the single, linear walkthrough from raw dataset to
  prediction, including what a run directory contains.
- [FEATURE_TEST_COMMANDS.md](FEATURE_TEST_COMMANDS.md) — every decoder × provider × augmentation ×
  optimizer × scheduler combination, for exercising the full config surface at once.
