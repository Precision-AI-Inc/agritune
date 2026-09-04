# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Synthetic tests for examples/datasets/prepare_cwfid.py — no live download."""

from __future__ import annotations

import csv
import importlib.util
from io import BytesIO
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "datasets" / "prepare_cwfid.py"


def _load_prepare_cwfid() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_cwfid_example", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_rgb(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _image_bytes(*, size: int = 2) -> bytes:
    return _png_rgb(np.full((size, size, 3), (12, 80, 20), dtype=np.uint8))


def _annotation_bytes() -> bytes:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 1] = (0, 255, 0)
    rgb[1, 0] = (255, 0, 0)
    return _png_rgb(rgb)


@pytest.fixture
def cwfid() -> ModuleType:
    return _load_prepare_cwfid()


def test_prepare_cwfid_converts_mocked_frames(
    tmp_path: Path, cwfid: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(url: str) -> bytes:
        if url.endswith("_image.png"):
            return _image_bytes()
        if url.endswith("_annotation.png"):
            return _annotation_bytes()
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(cwfid, "_fetch", fake_fetch)
    manifest_path = cwfid.prepare_cwfid(tmp_path, max_samples=2, size=2)
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    assert [row["sample_id"] for row in rows] == ["cwfid-001", "cwfid-002"]
    assert rows[0]["source"] == "cwfid"
    mask = np.array(Image.open(tmp_path / "masks" / "001.png"))
    assert mask.tolist() == [[0, 1], [2, 0]]


def test_annotation_to_class_mask_rejects_unknown_colours(cwfid: ModuleType) -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    with pytest.raises(ValueError, match="colours other than black/green/red"):
        cwfid._annotation_to_class_mask(_png_rgb(rgb), size=2)


def test_fetch_refuses_non_https_urls(cwfid: ModuleType) -> None:
    with pytest.raises(ValueError, match="non-HTTPS"):
        cwfid._fetch("http://example.test/images/001_image.png")


def test_prepare_cwfid_rejects_out_of_range_max_samples(tmp_path: Path, cwfid: ModuleType) -> None:
    with pytest.raises(ValueError, match="max_samples must be in"):
        cwfid.prepare_cwfid(tmp_path, max_samples=0, size=8)
    with pytest.raises(ValueError, match="max_samples must be in"):
        cwfid.prepare_cwfid(tmp_path, max_samples=61, size=8)


def test_prepare_cwfid_does_not_write_manifest_when_a_later_fetch_fails(
    tmp_path: Path, cwfid: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_fetch(url: str) -> bytes:
        del url
        calls["n"] += 1
        if calls["n"] > 2:
            raise cwfid.httpx.HTTPError("truncated download")
        return _image_bytes() if calls["n"] % 2 == 1 else _annotation_bytes()

    monkeypatch.setattr(cwfid, "_fetch", fake_fetch)
    with pytest.raises(cwfid.httpx.HTTPError, match="truncated download"):
        cwfid.prepare_cwfid(tmp_path, max_samples=2, size=2)
    assert not (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "images" / "001.png").is_file()


def test_main_reports_fetch_errors_on_stderr(
    tmp_path: Path, cwfid: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_prepare(*_args: object, **_kwargs: object) -> Path:
        raise cwfid.httpx.HTTPError("404")

    monkeypatch.setattr(cwfid, "prepare_cwfid", fail_prepare)
    exit_code = cwfid.main(["--output", str(tmp_path), "--max-samples", "1"])
    assert exit_code == 1
    assert "404" in capsys.readouterr().err


def test_main_full_flag_downloads_all_frames(
    tmp_path: Path, cwfid: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, int] = {}

    def fake_prepare(output_dir: Path, *, max_samples: int, size: int) -> Path:
        del output_dir, size
        seen["max_samples"] = max_samples
        return tmp_path / "manifest.csv"

    monkeypatch.setattr(cwfid, "prepare_cwfid", fake_prepare)
    assert cwfid.main(["--full", "--output", str(tmp_path)]) == 0
    assert seen["max_samples"] == cwfid._N_SOURCE_IMAGES
