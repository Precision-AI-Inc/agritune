# Feature-test commands (CWFID + hosted encoder)

Run from the **repo root** with the venv active. Requires:

- prepared data: `examples/datasets/cwfid/manifest.csv` (see `examples/datasets/README.md`)
- `.env` with `AGRITUNE_ENCODER_API_KEY` (`pip install -e ".[dotenv]"`)
- `agritune` on PATH (`pip install -e .`)

PowerShell snippets below. Cached training reuses features already in
`examples/datasets/cwfid/features` and does **not** call the encoder. Anything marked
**API** hits `https://embeddings.precision.ai/v1`.

Shared Hydra knobs (paste once per session):

```powershell
$cfg  = "precisionai/agritune/configs/config.yaml"
$data = "examples/datasets/cwfid"
$enc  = @(
  "encoder=remote"
  "encoder.base_url=https://embeddings.precision.ai/v1"
  "encoder.model=pai-embedding"
)
$dataArgs = @(
  "dataset.manifest_path=$data/manifest.csv"
  "dataset.num_classes=3"
  "feature_store_dir=$data/features"
  "run_root=$data/runs"
  "batch_size=4"
  "val_fraction=0.25"
  "seed=0"
  "trainer.max_epochs=1"
)
```

---

## 1. Dataset

```powershell
agritune dataset validate --manifest "$data/manifest.csv" --num-classes 3
agritune dataset inspect  --manifest "$data/manifest.csv"
```

Expect 24 samples, classes `{0,1,2}`, no issues.

---

## 2. Encoder **API**

```powershell
agritune encoder benchmark `
  --base-url https://embeddings.precision.ai/v1 `
  --model pai-embedding `
  --image-size 384 `
  --batch-sizes 1 2 4 `
  --concurrencies 1 2 `
  --num-requests 3
```

Fake encoder (no network), same CLI:

```powershell
agritune encoder benchmark --model fake-encoder --image-size 64 --batch-sizes 1 4 --concurrencies 1 2 --num-requests 3
```

---

## 3. Featurization **API** (resumable)

Skip `build` if `examples/datasets/cwfid/features` already has 24 entries.

```powershell
agritune features build --manifest "$data/manifest.csv" --store "$data/features" --base-url https://embeddings.precision.ai/v1 --model pai-embedding
agritune features build --manifest "$data/manifest.csv" --store "$data/features" --base-url https://embeddings.precision.ai/v1 --model pai-embedding   # second pass: skipped=24
agritune features verify  --store "$data/features"
agritune features inspect --store "$data/features"
agritune features clean   --store "$data/features"   # orphaned files only; safe on a healthy store
```

---

## 4. Train — cached features (no encoder calls)

Baseline (flat YAML):

```powershell
agritune train --config examples/segmentation/cwfid.yaml trainer.max_epochs=1 run_id=cwfid-cached-mlp
```

Every decoder (Hydra group swap; 1 epoch each):

```powershell
foreach ($dec in @("mlp_probe","token_fpn","aspp","ppm","segmenter","mask_former")) {
  agritune train --config $cfg @enc @dataArgs "decoder=$dec" "run_id=cwfid-dec-$dec"
}
```

Optimizers / schedulers / tracking / AMP / accumulation / clipping:

```powershell
agritune train --config $cfg @enc @dataArgs run_id=cwfid-sgd-cosine decoder=mlp_probe optimizer=sgd scheduler=cosine scheduler.total_steps=20 tracking=local trainer.precision=fp32 trainer.accumulation_steps=2 trainer.grad_clip_norm=1.0
agritune train --config $cfg @enc @dataArgs run_id=cwfid-adam-plateau optimizer=adam scheduler=plateau
agritune train --config $cfg @enc @dataArgs run_id=cwfid-warmup scheduler=linear_warmup scheduler.total_steps=20
agritune train --config $cfg @enc @dataArgs run_id=cwfid-poly scheduler=polynomial scheduler.total_steps=20
agritune train --config $cfg @enc @dataArgs run_id=cwfid-track-tb tracking=tensorboard
agritune train --config $cfg @enc @dataArgs run_id=cwfid-track-none tracking='"null"'
```

(`tracking=null` must be quoted — otherwise Hydra reads a YAML null.)

Early stopping + periodic checkpoints:

```powershell
agritune train --config $cfg @enc @dataArgs run_id=cwfid-early trainer.max_epochs=5 trainer.early_stopping_patience=1 trainer.checkpoint_every_n_steps=2
```

Resume (same `run_id`, more epochs):

```powershell
agritune train --config $cfg @enc @dataArgs run_id=cwfid-cached-mlp trainer.max_epochs=3
```

---

## 5. Train — augmentation × feature provider **API** where noted

| augmentation | feature_provider | encoder calls? |
|---|---|---|
| `none` (default) | `cached` | no |
| `offline` | `hybrid` | yes, only on cache miss of the augmented variant |
| `online` | `online` | yes, every batch |
| `hybrid` | `hybrid` | yes, on misses |
| `online` | `cached` | **rejected** |

Offline + hybrid (deterministic flip/rotate; writes new cache keys):

```powershell
agritune train --config $cfg @enc @dataArgs run_id=cwfid-aug-offline augmentation=offline feature_provider=hybrid trainer.max_epochs=1
```

Online + online provider:

```powershell
agritune train --config $cfg @enc @dataArgs run_id=cwfid-aug-online augmentation=online feature_provider=online trainer.max_epochs=1
```

Hybrid aug + hybrid provider:

```powershell
agritune train --config $cfg @enc @dataArgs run_id=cwfid-aug-hybrid augmentation=hybrid feature_provider=hybrid trainer.max_epochs=1
```

Must fail (invalid combo):

```powershell
agritune train --config $cfg @enc @dataArgs run_id=cwfid-should-fail augmentation=online feature_provider=cached
```

---

## 6. Evaluate / predict

`--encoder-revision` must be empty (hosted API has no revision; cache keys used `None`).

```powershell
$ckpt = "$data/runs/cwfid-cached-mlp/checkpoints/best.ckpt"
if (-not (Test-Path $ckpt)) { $ckpt = "$data/runs/cwfid-sanity-mlp_probe-feature_aug/checkpoints/best.ckpt" }

agritune evaluate `
  --manifest "$data/manifest.csv" --store "$data/features" --checkpoint $ckpt `
  --num-classes 3 --decoder mlp_probe --encoder-model pai-embedding --encoder-revision "" --batch-size 4

agritune evaluate `
  --manifest "$data/manifest.csv" --store "$data/features" --checkpoint $ckpt `
  --num-classes 3 --decoder mlp_probe --encoder-model pai-embedding --encoder-revision "" `
  --sample-ids cwfid-001 cwfid-002 cwfid-003

agritune predict `
  --manifest "$data/manifest.csv" --store "$data/features" --checkpoint $ckpt `
  --num-classes 3 --decoder mlp_probe --encoder-model pai-embedding --encoder-revision "" `
  --output "$data/predictions" --overlays --overlay-alpha 0.5
```

For a TokenFPN run, pass `--decoder token_fpn` and that run's checkpoint.

---

## 7. API (same services as the CLI)

```powershell
pip install uvicorn
python -m uvicorn precisionai.agritune.api.app:create_app --factory --port 8000
```

In another shell:

```powershell
curl.exe http://127.0.0.1:8000/docs
curl.exe -X POST http://127.0.0.1:8000/dataset/validate -H "Content-Type: application/json" -d "{\"manifest_path\": \"examples/datasets/cwfid/manifest.csv\", \"num_classes\": 3}"
curl.exe -X POST http://127.0.0.1:8000/features/inspect -H "Content-Type: application/json" -d "{\"store\": \"examples/datasets/cwfid/features\"}"
```

---

## 8. What you are exercising

| Area | Commands |
|---|---|
| Dataset adapter / validation / inspect | §1 |
| Remote encoder + gateway + rate limit + retries | §2, §3, §5 |
| Fake encoder | §2 |
| Feature store build / resume / verify / inspect / clean | §3 |
| Cached / online / hybrid providers | §4–5 |
| Offline / online / hybrid augmentation | §5 |
| Invalid config rejected up front | §5 last command |
| All 6 decoders | §4 loop |
| AdamW / Adam / SGD, cosine / warmup / poly / plateau | §4 |
| JSONL / TensorBoard / null tracking | §4 |
| Grad accumulation, clip, early stop, periodic ckpt, resume | §4 |
| Evaluate, restricted sample IDs, predict + overlays | §6 |
| FastAPI layer | §7 |
