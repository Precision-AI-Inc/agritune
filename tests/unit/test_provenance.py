# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.logging.provenance."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from precisionai.agritune.logging import provenance
from precisionai.agritune.logging.provenance import (
    EnvironmentInfo,
    GitInfo,
    RunDirectory,
    capture_environment_info,
    capture_git_info,
)


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_capture_git_info_outside_a_repo_returns_none_commit(tmp_path: Path) -> None:
    info = capture_git_info(tmp_path)
    assert info == GitInfo(commit=None, is_dirty=False)


def test_capture_git_info_returns_fallback_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> None:
        raise OSError("git executable not found")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    assert capture_git_info(tmp_path) == GitInfo(commit=None, is_dirty=False)


def test_capture_git_info_reports_commit_and_clean_tree(tmp_path: Path) -> None:
    _run_git(["init"], cwd=tmp_path)
    _run_git(["config", "user.email", "test@example.com"], cwd=tmp_path)
    _run_git(["config", "user.name", "Test"], cwd=tmp_path)
    (tmp_path / "file.txt").write_text("hello")
    _run_git(["add", "file.txt"], cwd=tmp_path)
    _run_git(["commit", "-m", "initial"], cwd=tmp_path)

    info = capture_git_info(tmp_path)
    assert info.commit is not None
    assert len(info.commit) == 40
    assert info.is_dirty is False


def test_capture_git_info_reports_dirty_tree(tmp_path: Path) -> None:
    _run_git(["init"], cwd=tmp_path)
    _run_git(["config", "user.email", "test@example.com"], cwd=tmp_path)
    _run_git(["config", "user.name", "Test"], cwd=tmp_path)
    (tmp_path / "file.txt").write_text("hello")
    _run_git(["add", "file.txt"], cwd=tmp_path)
    _run_git(["commit", "-m", "initial"], cwd=tmp_path)
    (tmp_path / "file.txt").write_text("changed")

    info = capture_git_info(tmp_path)
    assert info.is_dirty is True


def test_capture_environment_info_reports_python_and_torch_versions() -> None:
    info = capture_environment_info()
    assert isinstance(info, EnvironmentInfo)
    assert info.python_version
    assert info.torch_version
    assert "torch" in info.installed_packages or "pai-agritune" in info.installed_packages


def test_run_directory_creates_expected_subdirectories(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-001")
    assert run_dir.checkpoints_dir.is_dir()
    assert run_dir.predictions_dir.is_dir()
    assert run_dir.artifacts_dir.is_dir()


def test_run_directory_paths_point_under_the_run_id(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-001")
    assert run_dir.logs_path == tmp_path / "run-001" / "logs.jsonl"
    assert run_dir.metrics_path == tmp_path / "run-001" / "metrics.jsonl"


def test_write_provenance_writes_all_core_files(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-001")
    run_dir.write_provenance(
        config_original={"seed": 0},
        config_resolved={"seed": 0, "lr": 0.001},
        run_info={"seed": 0, "encoder_model": "fake"},
    )

    assert (run_dir.path / "config.original.yaml").is_file()
    assert (run_dir.path / "config.resolved.yaml").is_file()
    assert (run_dir.path / "run.json").is_file()
    assert (run_dir.path / "environment.json").is_file()
    assert (run_dir.path / "git.json").is_file()
    assert not (run_dir.path / "dataset.json").exists()
    assert not (run_dir.path / "encoder.json").exists()


def test_write_provenance_writes_dataset_and_encoder_info_when_given(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-001")
    run_dir.write_provenance(
        config_original={},
        config_resolved={},
        run_info={},
        dataset_info={"fingerprint": "abc"},
        encoder_info={"model": "fake"},
    )
    assert json.loads((run_dir.path / "dataset.json").read_text()) == {"fingerprint": "abc"}
    assert json.loads((run_dir.path / "encoder.json").read_text()) == {"model": "fake"}


def test_config_resolved_yaml_roundtrips(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-001")
    resolved = {"seed": 42, "optimizer": {"name": "adamw", "lr": 0.001}}
    run_dir.write_provenance(config_original={}, config_resolved=resolved, run_info={})
    loaded = yaml.safe_load((run_dir.path / "config.resolved.yaml").read_text())
    assert loaded == resolved


def test_run_json_roundtrips(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-001")
    run_info = {"seed": 7, "dataset_fingerprint": "d1", "split_fingerprint": "s1"}
    run_dir.write_provenance(config_original={}, config_resolved={}, run_info=run_info)
    loaded = json.loads((run_dir.path / "run.json").read_text())
    assert loaded == run_info


def test_environment_json_includes_captured_fields(tmp_path: Path) -> None:
    run_dir = RunDirectory(tmp_path, "run-001")
    run_dir.write_provenance(config_original={}, config_resolved={}, run_info={})
    loaded = json.loads((run_dir.path / "environment.json").read_text())
    assert "python_version" in loaded
    assert "torch_version" in loaded
