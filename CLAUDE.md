# AgriTune — Project Standards

Apply these standards when writing, reviewing, or refactoring code in AgriTune. This document extends the Precision AI Python project standards ([agri-template](https://github.com/Precision-AI-Inc/agri-template)) with rules specific to this repository's domain: training agricultural segmentation decoders on frozen features from a remote ViT encoder.

See [agritune_implementation_plan.md](agritune_implementation_plan.md) for the phased build-out this repository follows, and [README.md](README.md) for current status.

---

## Environment

- Python 3.10+, managed with `python -m venv .venv` (never conda)
- Install: `pip install -e ".[dev]"` → installs all deps including pre-commit
- Register hooks once: `pre-commit install`
- `torch` is a hard dependency; do not add CUDA-specific code paths that fail on CPU-only machines — CI runs CPU-only.

---

## The one architectural rule

```text
Trainer and Decoder must never directly call the encoder API.
```

Everything goes through `FeatureProvider` (`CachedFeatureProvider`, `OnlineFeatureProvider`, `HybridFeatureProvider`), which is the only thing that talks to `EncoderGateway`. Concretely:

- `precisionai.agritune.training` and `precisionai.agritune.tasks` must never import from `precisionai.agritune.encoder`.
- Swapping `features.provider` between `cached` / `online` / `hybrid` in config must never require a code change in the trainer or decoder.
- Reject any PR that violates this in review — it is the single most important invariant in the codebase.

---

## Project layout

```
precisionai/agritune/
  api/routes/       # thin FastAPI route handlers — delegate to services/
  augmentations/
    image/           # geometric + photometric image/mask transforms (deterministic, seeded)
    feature/          # feature-space augmentation (patch dropout, token masking, ...)
  cli/               # `agritune` entry point and subcommands (argparse)
  configs/           # Hydra config groups: dataset/, encoder/, augmentation/, feature_provider/,
                     # task/, decoder/, optimizer/, scheduler/, tracking/, plus root config.yaml
  data/              # dataset adapters (ManifestDataset, SegmentationDataset), split strategies
  encoder/           # EncoderBackend protocol, FakeEncoderBackend, RemoteEncoderBackend,
                     # EncoderGateway, batching, retry, rate_limiter, response validation
  features/          # FeatureProvider (Cached/Online/Hybrid/Prefetching), FeatureStore
                     # (Directory/Sharded), cache keys/fingerprints, manifest, resumable precompute,
                     # integrity (verify/clean)
  logging/           # configure_logging (stderr Rich), RedactingFilter, progress_iter, run provenance
  metrics/           # pure computation (segmentation metrics — no I/O)
  optimization/      # optimizer/scheduler registries
  schemas/           # EncoderFeatures, Sample, PreparedSample, AugmentationRecord, core Protocols
  services/          # orchestration (TrainingService, EvaluationService, FeatureService, ...) —
                     # both the CLI and the API call these, never duplicate logic
  tasks/segmentation/
    decoders/         # linear probe, TokenFPN
    visualization.py   # colorized prediction overlays
  tracking/          # Tracker protocol + Null/JSONL/TensorBoard/MLflow/WandB/Neptune/Comet + MultiTracker
  training/          # Trainer, evaluator, checkpointing, distributed, precision, state
  utils/
docs/                # Sphinx (HTML + LaTeX/PDF) + architecture/config/dataset/encoder/... guides
tests/               # unit/, integration/, distributed/, fixtures/ — mirrors package structure
examples/            # runnable config examples (segmentation/*.yaml), dataset samples
```

`precisionai` is a fixed, namespace-only top-level package (no logic, just an SPDX header) shared by every PAI Python project.

---

## Domain-specific rules

### Tensor shapes — never assume

Do not hardcode or assume any of the following anywhere outside a test fixture:

```text
D = 384                      # patch or CLS embedding dimension
N = 196, or a 14×14 grid     # number of patches / patch grid shape
square images
D_cls == D_patch
a fixed encoder revision string
```

The real encoder is served behind an OpenAI-SDK-compatible embeddings API with `native_resolution` support — patch grids are frequently non-square and dimensions are model-dependent. All of `patch_tokens`, `cls_tokens`, `patch_grid`, `image_sizes` on `EncoderFeatures` must stay dynamic. The API also exposes no queryable encoder revision — `RemoteEncoderBackend` fingerprints the encoder from the configured model alias plus the dimensions actually observed at runtime; do not assume a `models.list()`-style call can supply dimensions.

### Testing against the encoder

`FakeEncoderBackend` — not a live API key — is the basis for unit tests and CI. It must support configurable patch dimension, CLS dimension, patch grid, latency, and injectable failure modes (429, 500, timeout, malformed response). Never gate a unit test on `AGRITUNE_ENCODER_API_KEY` or similar being set; `-m integration` tests may use a real endpoint and are excluded from default CI/pre-commit runs.

### Secrets

Never log the encoder API key, the `Authorization` header, URL userinfo/query credentials, sensitive Hydra/dotlist overrides, or full request/response bodies. Call sites still must not put raw secrets into log messages; `configure_logging` also attaches `RedactingFilter` so those values (and exception-chain text that embeds them) are scrubbed before emit at every level, including DEBUG.

### Configuration

Config composition uses Hydra/OmegaConf (`configs/config.yaml` + config groups). Validate configuration before training starts and fail fast for known-invalid combinations (e.g. `augmentation=online` with `feature_provider=cached` unless a specific compatible behavior is implemented and documented) — do not let an invalid combination silently run.

### Reproducibility

Every training run must be reconstructable from its `runs/<run_id>/` directory alone: resolved config, dataset/split fingerprint, encoder fingerprint, feature cache fingerprint, git commit/dirty state, and RNG state (Python/NumPy/PyTorch CPU/CUDA) all get captured — this is a core requirement, not an optional enhancement. Do not add a training feature that can't be reproduced from a checkpoint plus its run directory.

---

## pyproject.toml — canonical config

```toml
[project]
dynamic = ["version"]   # version comes from the git tag via setuptools-scm, never hardcoded

[tool.setuptools_scm]
version_scheme = "no-guess-dev"
local_scheme = "no-local-version"
fallback_version = "0.0.0"

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E","W","F","I","UP","B","SIM","N","C90","D","PT","RUF","PL","ANN","PERF","S"]
ignore = [
    "E501",    # enforced by ruff-format
    "B008",    # FastAPI default-arg pattern
    "SIM108",  # ternary readability
    "D100","D104",          # module/package docstrings optional
    "D203","D213",          # pydocstyle conflicts — always ignore these two
    "PLR0913","PLR2004",    # arg count + magic values common in training/metrics/tests
    "ANN401",              # Any is allowed for genuinely dynamic types
    "S311",                # pseudo-random generators are intentional in scientific code
    "S104",                # binding to 0.0.0.0 is intentional for a configurable server host
]
[tool.ruff.lint.per-file-ignores]
"**/__init__.py" = ["F401"]
"tests/**"       = ["D","PLR","ANN","S"]
"docs/conf.py"   = ["E402","UP031","ANN","S"]
"precisionai/agritune/configs/**" = ["D"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"          # enforces NumPy docstring style

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.isort]
known-first-party = ["precisionai"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true  # formats code blocks inside docstrings

[tool.pytest.ini_options]
addopts = "--cov=precisionai --cov-report=term-missing --cov-fail-under=90"
markers = [
    "integration: marks tests that require external resources",
    "distributed: marks tests that require multi-process/multi-GPU coordination",
]

[tool.coverage.report]
fail_under = 90
exclude_lines = ["pragma: no cover","if __name__ == .__main__.:",
                 "raise ImportError","except ImportError",
                 "^\\s*\\.\\.\\.\\s*$"]  # Protocol stub bodies

[tool.pyright]
pythonVersion = "3.10"
typeCheckingMode = "standard"
reportMissingImports = false
reportMissingModuleSource = false
```

---

## pre-commit

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks: [check-added-large-files (--maxkb=1800), check-yaml, check-json,
            check-toml, end-of-file-fixer, trailing-whitespace,
            check-merge-conflict, detect-private-key, debug-statements,
            fix-byte-order-marker]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.17
    hooks: [ruff (--fix), ruff-format]

  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: python -m pyright
        language: system
        types: [python]
        pass_filenames: false

  - repo: local
    hooks:
      - id: pytest
        entry: python -m pytest
        language: system
        args: [tests/, -q, --tb=short, --no-header, -m, "not integration and not distributed"]
        always_run: true
```

---

## Code style

### License header
- Every `.py` file starts with a two-line SPDX header, followed by a blank line before the module docstring (if any):
  ```python
  # Copyright 2026 Precision AI
  # SPDX-License-Identifier: Apache-2.0
  ```
- No "CONFIDENTIAL" or proprietary banners — this codebase is Apache 2.0 licensed; any such banner is a bug.

### Imports
- All imports at the **top of the file** — never inside functions
- Optional/unavailable deps (e.g. `mlflow`, `wandb`): module-level `try/except ImportError` with a flag variable
- No silent fallbacks unless mathematically equivalent (document why)
- `__init__.py` re-exports only — no logic, no comments between import blocks
- `__all__` must be **alphabetically sorted**

```python
# optional dep pattern
try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# inside the class/function that needs it
if not _WANDB_AVAILABLE:
    raise ImportError("wandb is required: pip install pai-agritune[tracking]") from None
```

### Docstrings
- **NumPy style** on all public functions, classes, and modules
- Module docstrings: plain English, no prefixes (no P0/P1/P2, no "original", no labels)
- One-line summary in **imperative mood** ("Compute …", "Return …", "Save …")
- Blank line between summary and extended description (D205)
- Document tensor shapes and dtypes explicitly in `Parameters`/`Returns` (e.g. `patch_tokens : torch.Tensor of shape (B, N, D)`)

### Type hints
- Required on **all** function signatures (public and private) — enforced by ruff (`ANN`) and pyright
- Use `X | Y` union syntax (Python 3.10+), not `Union[X, Y]` or `(X, Y)` in isinstance
- Use `Any` from `typing` for genuinely dynamic types — don't use `object` when methods will be called on it
- pyright config lives in `[tool.pyright]` in `pyproject.toml` — never pass type-checker flags inline

### Comments
- Only when the **why** is non-obvious
- No section-divider comments that describe what the code already says
- No cryptic labels, no TODO/FIXME without a ticket reference

### Exception handling
- Always `raise X from err` or `raise X from None` inside `except` blocks (B904)

### General
- `len()` returns `int` — never `int(len(...))`
- Loop variables not used in the body → rename to `_`
- `assert a and b` in tests → split into separate asserts
- `@pytest.fixture` not `@pytest.fixture()`
- `pytest.raises` always includes `match=` parameter
- No `# type: ignore[attr-defined]` when `Any` already covers the attribute

---

## Docs (Sphinx)

```
docs/
  conf.py          # version from git tag, logo, LaTeX/fancyhdr, enumitem fix
  index.rst        # toctree: readme, architecture, configuration, datasets, encoder,
                   # feature-caching, augmentation, segmentation, reproducibility,
                   # extending-agritune, modules
  modules.rst      # autodoc for all public submodules
  architecture.md, configuration.md, datasets.md, encoder.md, feature-caching.md,
  augmentation.md, segmentation.md, reproducibility.md, extending-agritune.md
  requirements.txt # sphinx, sphinx-rtd-theme, myst-parser, sphinx-autodoc-typehints
  Makefile         # make html | make latexpdf | make clean
  assets/logo.png
```

- `conf.py` copies root `README.md` → `docs/readme.md` at build time
- Version read from `git describe --tags --exact-match`, falls back to `0.0.0`

---

## Testing

- Mirror package structure under `tests/unit/`; integration tests (fake-encoder end-to-end pipelines) live in `tests/integration/`; multi-process/DDP tests live in `tests/distributed/`; shared fixtures live in `tests/fixtures/` and `tests/conftest.py`
- 90% coverage hard minimum — enforced by pytest and pre-commit, on the unit-test subset
- No mocks for the filesystem/feature store unless truly unavoidable — use `tmp_path`
- The encoder is the one exception: unit tests use `FakeEncoderBackend`, never a live API
- Integration/distributed tests marked `@pytest.mark.integration` / `@pytest.mark.distributed` and excluded from default runs (`-m "not integration and not distributed"`)

---

## Repository scaffolding

Every PAI Python project repository, in addition to the source layout above, ships with:

```
.gitattributes              # normalize line endings to LF; declare binary assets
CHANGELOG.md                # Keep a Changelog format + Semantic Versioning
CODE_OF_CONDUCT.md          # Contributor Covenant v2.1
MANIFEST.in                 # sdist packaging — include docs/license/changelog, exclude tests/examples
.github/
  dependabot.yml            # weekly pip + github-actions update checks
  ISSUE_TEMPLATE/
    bug_report.yml
    feature_request.yml
    config.yml               # blank_issues_enabled: false + link to SECURITY.md
  PULL_REQUEST_TEMPLATE.md
  workflows/
    ci.yml                   # dispatches to pre-commit.yml + unit-test.yml
    pre-commit.yml
    unit-test.yml
    release.yml               # tag push → build, PyPI publish, GitHub Release
```

These are not optional extras — treat them as part of the standard layout when auditing this project.

## Releasing

- Versions are **never hardcoded** — `setuptools-scm` derives the package version from the current git tag. Tag with `vX.Y.Z`.
- Update `CHANGELOG.md` under `[Unreleased]` as you go; move those entries to a new `## [X.Y.Z] - YYYY-MM-DD` section when cutting a release.
- Pushing a `v*.*.*` tag triggers `.github/workflows/release.yml`: runs the full test matrix, builds the sdist/wheel, verifies the built version matches the tag, publishes to PyPI via trusted publishing (`id-token: write`, no stored API tokens), and creates a GitHub Release with generated notes.
- CI checkout steps that build from git history (release builds, docs version detection) must use `fetch-depth: 0` — a shallow checkout has no tags for `setuptools-scm` or `git describe` to find.

---

## What to avoid

| Pattern | Instead |
|---|---|
| `int(len(x))` | `len(x)` |
| `isinstance(x, (A, B))` | `isinstance(x, A \| B)` |
| `assert a and b` (tests) | two separate asserts |
| `@pytest.fixture()` | `@pytest.fixture` |
| `pytest.raises(ValueError)` | `pytest.raises(ValueError, match="…")` |
| `raise X` inside except | `raise X from None` or `raise X from err` |
| Deferred imports inside functions | Module-level try/except with flag |
| Cryptic prefixes (P0–P4, "original") | Plain descriptive names |
| `# type: ignore[attr-defined]` on `Any` | Remove — redundant |
| `# noqa` suppression | Fix the underlying issue |
| Silent fallback for optional dep | Raise `ImportError` with install hint |
| Hardcoded `D=384` / `14x14` grid / square image assumption | Read shapes from `EncoderFeatures` dynamically |
| Trainer/Decoder importing `precisionai.agritune.encoder` | Go through `FeatureProvider` |
| Unit test requiring a live encoder API key | Use `FakeEncoderBackend`; mark real-API tests `@pytest.mark.integration` |
