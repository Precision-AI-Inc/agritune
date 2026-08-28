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

This is a core v1 requirement, not an optional enhancement — see `agritune_implementation_plan.md`
§16.

## Checkpoints carry more than weights

A checkpoint saves decoder state, optimizer state, scheduler state, gradient scaler, epoch, micro
step, global optimizer step, best metric, and full RNG state (Python, NumPy, PyTorch CPU, CUDA) —
so training resumes bit-for-bit, not just "close enough." Resume warns or fails when critical
configuration differs (encoder revision changed, class count changed, decoder architecture
changed) — see `agritune_implementation_plan.md` §14.

## Feature cache fingerprint

Cached features are invalidated whenever anything that affects them changes: sample ID, image
hash, augmentation configuration/seed, encoder model, encoder revision, encoder preprocessing, and
the feature schema version. See [feature-caching.md](feature-caching.md).

_This document expands as the checkpointing (Phase 11) and provenance (Phase 13) subsystems land._
