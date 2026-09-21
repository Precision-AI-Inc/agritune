# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.api.paths.resolve_under_root and api.config.get_api_root."""

from pathlib import Path

import pytest

from precisionai.agritune.api.config import get_api_root
from precisionai.agritune.api.paths import resolve_under_root


def test_get_api_root_defaults_to_the_current_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGRITUNE_API_ROOT", raising=False)
    assert get_api_root() == Path.cwd().resolve()


def test_get_api_root_reads_the_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGRITUNE_API_ROOT", str(tmp_path))
    assert get_api_root() == tmp_path.resolve()


def test_resolve_under_root_accepts_a_relative_path_inside_root(tmp_path: Path) -> None:
    (tmp_path / "manifest.csv").touch()
    resolved = resolve_under_root(tmp_path, "manifest.csv", field_name="manifest_path")
    assert resolved == tmp_path.resolve() / "manifest.csv"


def test_resolve_under_root_accepts_an_absolute_path_inside_root(tmp_path: Path) -> None:
    nested = tmp_path / "features"
    resolved = resolve_under_root(tmp_path, str(nested), field_name="store")
    assert resolved == nested.resolve()


def test_resolve_under_root_accepts_root_itself(tmp_path: Path) -> None:
    assert resolve_under_root(tmp_path, ".", field_name="store") == tmp_path.resolve()


def test_resolve_under_root_rejects_a_dotdot_segment_that_escapes_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest_path"):
        resolve_under_root(tmp_path, "../escaped.csv", field_name="manifest_path")


def test_resolve_under_root_rejects_an_absolute_path_outside_root(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "checkpoint.ckpt"
    with pytest.raises(ValueError, match="checkpoint_path"):
        resolve_under_root(tmp_path, str(outside), field_name="checkpoint_path")
