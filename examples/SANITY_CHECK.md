# CWFID sanity-check pipeline

End-to-end AgriTune run on a **real public agricultural segmentation dataset** (CWFID) with the
hosted `pai-embedding` encoder. Run from the repository root after `pip install -e ".[dev,dotenv]"`
and a `.env` containing `AGRITUNE_ENCODER_API_KEY`.

PowerShell and bash are shown where they differ; otherwise the commands are the same.

To exercise **every** CLI/config feature (all decoders, providers, augmentations, schedulers)
on this dataset, see [FEATURE_TEST_COMMANDS.md](FEATURE_TEST_COMMANDS.md).

## 0. Prepare the dataset

Downloads 24 CWFID frames, converts RGB crop/weed maps to class-index masks, resizes to 384×384,
writes `examples/datasets/cwfid/manifest.csv`. See [datasets/README.md](datasets/README.md).

```bash
python examples/datasets/prepare_cwfid.py --output examples/datasets/cwfid --max-samples 24 --size 384
```

Full 60-image set: `--full` (equivalent to `--max-samples 60`).

## 1. Validate and inspect

```bash
agritune dataset validate --manifest examples/datasets/cwfid/manifest.csv --num-classes 3
agritune dataset inspect --manifest examples/datasets/cwfid/manifest.csv
```

Expect 24 samples, 3 classes (background / crop / weed), no issues.

## 2. (Optional) Benchmark the hosted encoder

```bash
agritune encoder benchmark --base-url https://embeddings.precision.ai/v1 --model pai-embedding --image-size 384 --batch-sizes 1 2 4 --concurrencies 1 2 --num-requests 3
```

## 3. Precompute features

```bash
agritune features build --manifest examples/datasets/cwfid/manifest.csv --store examples/datasets/cwfid/features --base-url https://embeddings.precision.ai/v1 --model pai-embedding
agritune features verify --store examples/datasets/cwfid/features
agritune features inspect --store examples/datasets/cwfid/features
```

Re-running `features build` is resumable: already-cached sample IDs are skipped.

## 4. Train

```bash
agritune train --config examples/segmentation/cwfid.yaml
```

Hydra equivalent (same packaged groups, explicit remote encoder):

```bash
agritune train --config precisionai/agritune/configs/config.yaml \
    encoder=remote \
    encoder.base_url=https://embeddings.precision.ai/v1 \
    encoder.model=pai-embedding \
    dataset.manifest_path=examples/datasets/cwfid/manifest.csv \
    dataset.num_classes=3 \
    feature_store_dir=examples/datasets/cwfid/features \
    run_root=examples/datasets/cwfid/runs \
    run_id=cwfid-sanity-hydra \
    decoder=mlp_probe \
    trainer.max_epochs=3 \
    batch_size=4 \
    val_fraction=0.25
```

Swap the decoder without touching the rest of the stack:

```bash
agritune train --config examples/segmentation/cwfid.yaml run_id=cwfid-tokenfpn decoder_name=token_fpn
```

Resume (same `run_root`/`run_id`, more epochs):

```bash
agritune train --config examples/segmentation/cwfid.yaml trainer.max_epochs=6
```

## 5. Evaluate

`--encoder-revision` must be empty: the hosted API does not expose a revision, and the feature
store was keyed with `revision=None`.

```bash
agritune evaluate \
    --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --checkpoint examples/datasets/cwfid/runs/cwfid-sanity-mlp_probe-feature_aug/checkpoints/best.ckpt \
    --num-classes 3 \
    --decoder mlp_probe \
    --encoder-model pai-embedding \
    --encoder-revision "" \
    --batch-size 4
```

## 6. Predict (class maps + overlays)

```bash
agritune predict \
    --manifest examples/datasets/cwfid/manifest.csv \
    --store examples/datasets/cwfid/features \
    --checkpoint examples/datasets/cwfid/runs/cwfid-sanity-mlp_probe-feature_aug/checkpoints/best.ckpt \
    --num-classes 3 \
    --decoder mlp_probe \
    --encoder-model pai-embedding \
    --encoder-revision "" \
    --output examples/datasets/cwfid/predictions \
    --overlays \
    --overlay-alpha 0.5
```

Predictions: `examples/datasets/cwfid/predictions/{sample_id}.png`  
Overlays: `examples/datasets/cwfid/predictions/{sample_id}_overlay.png`

## 7. What a successful run directory contains

```text
examples/datasets/cwfid/runs/cwfid-sanity-mlp_probe-feature_aug/
├── config.original.yaml
├── config.resolved.yaml      # API key redacted
├── run.json
├── dataset.json
├── encoder.json
├── environment.json
├── git.json
├── metrics.jsonl
└── checkpoints/
    ├── last.ckpt
    └── best.ckpt
```
