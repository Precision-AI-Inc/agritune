# Reproducibility

Every AgriTune run must be reconstructable from its `runs/<run_id>/` directory alone.

```text
runs/<run_id>/
├── config.original.yaml
├── config.resolved.yaml
├── run.json
├── dataset.json
├── encoder.json
├── environment.json
├── git.json
├── logs.jsonl
├── metrics.jsonl
├── checkpoints/
├── predictions/
└── artifacts/
```

Captured provenance:

```text
Git commit, dirty state
Python version, PyTorch version, CUDA version, installed packages
hostname, GPU model
seed
dataset fingerprint, train/val/test split fingerprint
encoder model, encoder revision (see docs/encoder.md — opaque unless pinned), encoder preprocessing
feature cache fingerprint
```

This is a core v1 requirement, not an optional enhancement. Provenance is written twice: once
immediately after fingerprints are computed, *before* training starts, and again when the run
finishes with the final epoch/step/best-metric filled in. The early write means a run that crashes
or is killed mid-training still leaves a reconstructable record — including for diagnosing a
`CheckpointMismatchError` on resume, which otherwise has no record of what config an existing
checkpoint was fingerprinted against.

## Checkpoints carry more than weights

A checkpoint saves decoder state, optimizer state, scheduler state, gradient scaler, epoch, batch
position within the epoch, micro step, global optimizer step, best metric, and full RNG state
(Python, NumPy, PyTorch CPU, CUDA) —
so training resumes bit-for-bit, not just "close enough." Resume warns or fails when critical
configuration differs (encoder revision changed, class count changed, decoder architecture
changed). `last.ckpt`/`best.ckpt` are always written;
periodic checkpoints ranked by the validation metric are also kept, up to a configurable
`checkpoint_top_k` (`0` keeps every one).

Set `trainer.strict_resume: false` to log a warning instead of raising `CheckpointMismatchError` on
a fingerprint mismatch, and resume anyway — only after confirming the mismatch is benign. This only
bypasses the fingerprint *check*: decoder/optimizer state still loads as-is, so a genuine
architecture or dataset change resumed this way fails later with a `state_dict` shape-mismatch
error instead, or worse, loads silently incorrect state. `strict_resume` itself is excluded from
the config fingerprint, so toggling it never causes a mismatch on its own.

## Feature cache fingerprint

Cached features are invalidated whenever anything that affects them changes: sample ID, image
hash, augmentation configuration/seed, encoder model, encoder revision, encoder preprocessing, and
the feature schema version. See [feature-caching.md](feature-caching.md).
