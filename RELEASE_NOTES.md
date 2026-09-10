# AgriTune: train segmentation decoders on a frozen ViT encoder, end to end

*Precision AI · release note*

AgriTune trains and evaluates agricultural image-segmentation models without
training the encoder itself. A lightweight decoder learns on top of features
from a remote, frozen ViT encoder — everything from a raw dataset manifest to
a checkpointed, reproducible training run is built and working. This is the
first release: nothing has shipped before it, so this note covers the whole
feature set, not just what's new.

```
$ agritune train --config examples/segmentation/cwfid.yaml
...
run directory: runs/2026-09-10_141203
final epoch: 12
train metrics: {'loss': 0.184, 'mean_iou': 0.71}
val metrics: {'loss': 0.201, 'mean_iou': 0.68}
```

## Why it matters

Training or fine-tuning a full vision transformer needs GPU budget and data
most agricultural teams don't have on day one. AgriTune gets you a working
crop/weed segmentation model by training a small decoder on top of a hosted
encoder's features instead — cheaper to iterate on, and every run is
reproducible enough to trust the result.

## What it unlocks

**Any dataset, any encoder shape.** Patch count, grid shape, and embedding
size are read from the encoder's response at runtime, not hardcoded —
non-square patch grids and different model sizes just work. Two example
datasets are ready to run: CWFID (60 field images) and PhenoBench (1,407
train / 772 val real UAV sugar-beet images).

**Three ways to get features, no code change to switch.** Precompute once
and train many times (`CachedFeatureProvider`), call the encoder live during
training (`OnlineFeatureProvider`/`HybridFeatureProvider`), or overlap
encoding with the GPU training step over a background queue
(`PrefetchingFeatureProvider`) so network latency stops blocking training.
Precomputing features now runs in fixed-size chunks, so memory use stays flat
regardless of manifest size.

**Augmentation in both image space and feature space.** Geometric and
photometric image transforms, plus agriculture-specific ones: sensor channel
dropout, ground-sample-distance jitter, vegetation-index-balance jitter, and
a seasonal/growth-stage hue shift. A hybrid mode reuses a deterministic
offline variant most of the time and mixes in fresh online augmentation for
a configurable fraction of samples. Feature-space augmentation (patch
dropout, token masking, feature noise, CLS/channel dropout) runs as an
optional training-only step.

**Six decoder architectures**: an MLP probe (linear by default, deeper on
request), TokenFPN, ASPP (DeepLabv3-style), Pyramid Pooling (PSPNet-style), a
Segmenter-style mask transformer, and a MaskFormer-style query-based decoder.

**Training built to be reproduced, not just run.** AMP, gradient
accumulation, checkpoint/exact-resume, early stopping, per-class
precision/recall alongside mIoU/Dice, and validation loss usable as the
checkpoint signal. Every run directory captures the resolved config,
dataset/split fingerprint, encoder fingerprint, feature-cache fingerprint,
git commit state, and full RNG state — a run is reconstructable from its
directory alone. Six tracking backends (JSONL, TensorBoard, MLflow, Weights &
Biases, Neptune, Comet), plus a no-op default and fan-out to several at once.

**Configure by composing, not editing code.** Hydra config groups for
dataset, encoder, augmentation, feature provider, task, decoder, optimizer,
scheduler, and tracking. `agritune config init`/`agritune dataset init`
scaffold a starting config and manifest. Invalid combinations (e.g. online
augmentation with a cached-only feature provider) fail before training
starts, not partway through.

**A CLI and an API, same code underneath.** Every CLI command — dataset
validate/inspect, features build/verify/inspect/clean, train, evaluate,
predict, encoder benchmark — has a matching FastAPI route calling the exact
same service function.

**Secrets never reach the logs.** API keys, Authorization headers, sensitive
config overrides, and URL credentials are redacted before anything is
logged, at every log level.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
agritune train --config examples/segmentation/cwfid.yaml
```

Anything beyond a quick look at the CLI needs a hosted encoder API key
(`AGRITUNE_ENCODER_API_KEY`, or a `.env` file) — AgriTune consumes features
from that API, it doesn't run the encoder itself.

---

# Hardening AgriTune ahead of its public release

*Precision AI · release note*

A pre-release read-through of the repo turned up one real security gap and a handful of
packaging/documentation issues. All of them are fixed now; this note covers what changed and why
it mattered.

## The one to actually stop and fix

Every route on AgriTune's optional FastAPI server took a raw filesystem path straight from the
request body — a checkpoint to load, a directory to write predictions into, a config file to read
— with no check that the path stayed anywhere sensible. A request could reach any file the server
process could read or write, via an absolute path or a `../` that climbed out of wherever the
caller was expected to stay.

Every path-shaped field (`manifest_path`, `store`, `checkpoint_path`, `output_dir`, `config_path`,
`augmentation_config_path`) is now resolved and contained to an API root (`AGRITUNE_API_ROOT`,
defaulting to the server's working directory) before it touches disk; anything that would escape
that root comes back as a 400 instead of being followed. The CLI is unaffected — it's a trusted
local tool and keeps full filesystem access, same as before.

## Packaging and legal cleanup

- **Dependencies.** `requirements.txt` used to hard-pin a CUDA build of `torch` that fails outright
  on any machine without that exact CUDA toolchain, CPU-only boxes included — it's a plain
  `torch>=2.2` now, with the CPU-only install path spelled out for Linux, where PyPI's default
  wheel otherwise pulls in several gigabytes of unused CUDA runtime. `.github/dependabot.yml` now
  groups weekly Python bumps into one PR instead of one per package.
- **License and Code of Conduct.** `LICENSE.md` now matches the canonical Apache 2.0 text exactly,
  and `CODE_OF_CONDUCT.md` was missing a whole section — the Enforcement Guidelines (warning →
  temporary ban → permanent ban) from Contributor Covenant v2.1 — restored verbatim.

## Finding your way in

The README used to open with an internal build-status paragraph and an installation guide before
saying anything about what to actually run. It now leads with a Quick Start that walks the packaged
CWFID crop/weed example start to finish — prepare the dataset, precompute features, train,
evaluate — on real data, with a `fake-encoder` fallback for anyone without hosted-encoder access
yet:

```
$ python examples/datasets/prepare_cwfid.py --output examples/datasets/cwfid --max-samples 24 --size 384
$ agritune features build --manifest examples/datasets/cwfid/manifest.csv --store examples/datasets/cwfid/features \
    --base-url https://embeddings.precision.ai/v1 --model pai-embedding
$ agritune train --config examples/segmentation/cwfid.yaml
```

Running the FastAPI server now gets its own install step (`pip install "pai-agritune[api]"`, which
actually includes `uvicorn` now), and the stale internal "Status" section is gone. Two of the
example docs (`SANITY_CHECK.md`, `FEATURE_TEST_COMMANDS.md`) also pointed evaluate/predict at a
checkpoint path the packaged training config never actually produces — fixed to the real one.

---

*Precision AI · September 2026*
