# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Run provenance capture.

Every training run gets a ``runs/<run_id>/`` directory capturing everything needed to reproduce
it — this is a core v1 requirement, not an optional enhancement.
"""

import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class GitInfo:
    """Git state at the time a run started.

    Attributes
    ----------
    commit : str | None
        The current commit hash, or ``None`` if not run inside a git repository (or ``git`` is
        unavailable).
    is_dirty : bool
        Whether the working tree has uncommitted changes.
    """

    commit: str | None
    is_dirty: bool


def capture_git_info(repo_root: str | Path | None = None) -> GitInfo:
    """Capture the current git commit and dirty state.

    Parameters
    ----------
    repo_root : str | Path | None, optional
        Directory to run ``git`` in; defaults to the current working directory.

    Returns
    -------
    GitInfo
        ``GitInfo(commit=None, is_dirty=False)`` if this isn't a git repository, ``git`` isn't
        installed, or the command otherwise fails — provenance capture must never abort a run.
    """
    cwd = str(repo_root) if repo_root is not None else "."
    # git is a trusted system tool, not attacker-controlled input — a partial path is intentional
    # (same precedent as docs/conf.py's version detection).
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
            check=False,
        )
        if commit_result.returncode != 0:
            return GitInfo(commit=None, is_dirty=False)
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
            check=False,
        )
        return GitInfo(commit=commit_result.stdout.strip(), is_dirty=bool(status_result.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        return GitInfo(commit=None, is_dirty=False)


@dataclass
class EnvironmentInfo:
    """The software/hardware environment a run executed in.

    Attributes
    ----------
    python_version : str
    torch_version : str
    cuda_version : str | None
        ``None`` when PyTorch was built without CUDA support.
    hostname : str
    gpu_model : str | None
        ``None`` when no CUDA device is available.
    installed_packages : dict[str, str]
        Every installed distribution name mapped to its version.
    """

    python_version: str
    torch_version: str
    cuda_version: str | None
    hostname: str
    gpu_model: str | None
    installed_packages: dict[str, str]


def capture_environment_info() -> EnvironmentInfo:
    """Capture the current Python/PyTorch/CUDA/hostname/GPU/installed-package environment."""
    installed_packages = {dist.name: dist.version for dist in importlib.metadata.distributions() if dist.name}
    gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None  # pragma: no cover
    return EnvironmentInfo(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        hostname=platform.node(),
        gpu_model=gpu_model,
        installed_packages=installed_packages,
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class RunDirectory:
    """Creates and populates ``runs/<run_id>/`` — the reproducibility artifact for one run.

    Parameters
    ----------
    root : str | Path
        Directory runs are stored under (typically ``"runs"``).
    run_id : str
        Unique identifier for this run; becomes the subdirectory name.
    """

    def __init__(self, root: str | Path, run_id: str) -> None:
        self.run_id = run_id
        self.path = Path(root) / run_id
        for subdir in ("checkpoints", "predictions", "artifacts"):
            (self.path / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for training checkpoints (``last.ckpt``, ``best.ckpt``, ...)."""
        return self.path / "checkpoints"

    @property
    def predictions_dir(self) -> Path:
        """Directory for prediction/visualization outputs (``agritune predict``)."""
        return self.path / "predictions"

    @property
    def artifacts_dir(self) -> Path:
        """Directory for miscellaneous logged artifacts."""
        return self.path / "artifacts"

    @property
    def logs_path(self) -> Path:
        """Path to this run's structured log file."""
        return self.path / "logs.jsonl"

    @property
    def metrics_path(self) -> Path:
        """Path to this run's JSONL metrics file (for :class:`~precisionai.agritune.tracking.jsonl.JSONLTracker`)."""
        return self.path / "metrics.jsonl"

    def write_provenance(
        self,
        *,
        config_original: dict[str, Any],
        config_resolved: dict[str, Any],
        run_info: dict[str, Any],
        dataset_info: dict[str, Any] | None = None,
        encoder_info: dict[str, Any] | None = None,
    ) -> None:
        """Write every provenance file this run directory is responsible for.

        Parameters
        ----------
        config_original : dict[str, Any]
            The configuration exactly as supplied (before defaults/overrides resolve).
        config_resolved : dict[str, Any]
            The fully resolved configuration actually used.
        run_info : dict[str, Any]
            Run-level metadata (e.g. seed, start time, dataset/split/encoder/feature-cache
            fingerprints) — shape is caller-defined since it varies by task.
        dataset_info : dict[str, Any] | None, optional
            Dataset-specific metadata, written to ``dataset.json`` if given.
        encoder_info : dict[str, Any] | None, optional
            Encoder-specific metadata, written to ``encoder.json`` if given.
        """
        _write_yaml(self.path / "config.original.yaml", config_original)
        _write_yaml(self.path / "config.resolved.yaml", config_resolved)
        _write_json(self.path / "run.json", run_info)
        _write_json(self.path / "environment.json", asdict(capture_environment_info()))
        _write_json(self.path / "git.json", asdict(capture_git_info()))
        if dataset_info is not None:
            _write_json(self.path / "dataset.json", dataset_info)
        if encoder_info is not None:
            _write_json(self.path / "encoder.json", encoder_info)
