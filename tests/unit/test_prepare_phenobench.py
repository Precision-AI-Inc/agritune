# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Synthetic tests for examples/datasets/prepare_phenobench.py — no live download."""

from __future__ import annotations

import csv
import importlib.util
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "datasets" / "prepare_phenobench.py"


def _load_prepare_phenobench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_phenobench_example", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_rgb(*, size: int = 2, color: tuple[int, int, int] = (12, 80, 20)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (size, size), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _png_labels(labels: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(labels.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _write_archive(path: Path, *, stems: dict[str, list[str]], labels: np.ndarray) -> None:
    image_bytes = _png_rgb(size=labels.shape[1])
    semantics_bytes = _png_labels(labels)
    with zipfile.ZipFile(path, "w") as archive:
        for split, names in stems.items():
            for name in names:
                archive.writestr(f"PhenoBench/{split}/images/{name}.png", image_bytes)
                archive.writestr(f"PhenoBench/{split}/semantics/{name}.png", semantics_bytes)


@pytest.fixture
def phenobench() -> ModuleType:
    return _load_prepare_phenobench()


def test_prepare_phenobench_converts_synthetic_splits_and_remaps_partial_labels(
    tmp_path: Path, phenobench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "PhenoBench-v110.zip"
    labels = np.array([[0, 1], [3, 4]], dtype=np.uint8)
    _write_archive(
        zip_path,
        stems={
            "train": ["2021-05-20_001_plotA"],
            "val": ["2021-06-01_002_plotB"],
        },
        labels=labels,
    )
    monkeypatch.setattr(phenobench, "_download_zip", lambda _path: None)

    output = tmp_path / "out"
    manifest_path = phenobench.prepare_phenobench(
        output, zip_path=zip_path, splits=["train", "val"], max_samples=2, size=2
    )

    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    assert [row["sample_id"] for row in rows] == [
        "phenobench-train-2021-05-20_001_plotA",
        "phenobench-val-2021-06-01_002_plotB",
    ]
    assert rows[0]["field_id"] == "plotA"
    assert rows[0]["capture_date"] == "2021-05-20"
    assert rows[1]["split"] == "val"
    train_mask = np.array(Image.open(output / "masks" / "train_2021-05-20_001_plotA.png"))
    assert train_mask.tolist() == [[0, 1], [1, 2]]
    assert set(np.unique(train_mask).tolist()) <= {0, 1, 2}


def test_semantics_to_class_mask_rejects_unexpected_labels(phenobench: ModuleType) -> None:
    data = _png_labels(np.array([[0, 5]], dtype=np.uint8))
    with pytest.raises(ValueError, match="label id outside"):
        phenobench._semantics_to_class_mask(data, size=2)


def test_prepare_phenobench_rejects_empty_and_unsupported_splits(tmp_path: Path, phenobench: ModuleType) -> None:
    with pytest.raises(ValueError, match="splits must not be empty"):
        phenobench.prepare_phenobench(tmp_path, zip_path=tmp_path / "x.zip", splits=[], max_samples=1, size=8)
    with pytest.raises(ValueError, match="unsupported split"):
        phenobench.prepare_phenobench(tmp_path, zip_path=tmp_path / "x.zip", splits=["test"], max_samples=1, size=8)


def test_prepare_phenobench_rejects_non_positive_bounds(tmp_path: Path, phenobench: ModuleType) -> None:
    with pytest.raises(ValueError, match="max_samples must be positive"):
        phenobench.prepare_phenobench(tmp_path, zip_path=tmp_path / "x.zip", splits=["train"], max_samples=0, size=8)
    with pytest.raises(ValueError, match="size must be positive"):
        phenobench.prepare_phenobench(tmp_path, zip_path=tmp_path / "x.zip", splits=["train"], max_samples=1, size=0)


def test_download_zip_skips_network_when_cached_size_matches(
    tmp_path: Path, phenobench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "PhenoBench-v110.zip"
    zip_path.write_bytes(b"cached-archive")
    monkeypatch.setattr(phenobench, "_ZIP_SIZE_BYTES", zip_path.stat().st_size)

    def _fail_stream(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cached archive should not be re-downloaded")

    monkeypatch.setattr(phenobench.httpx, "stream", _fail_stream)
    phenobench._download_zip(zip_path)
    assert zip_path.read_bytes() == b"cached-archive"


class _FakeStream:
    def __init__(self, payload: bytes, error: Exception | None = None) -> None:
        self.headers = {"content-length": str(len(payload))}
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int = 1) -> object:
        del chunk_size
        yield self._payload
        if self._error is not None:
            raise self._error

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_download_zip_deletes_partial_file_on_checksum_mismatch(
    tmp_path: Path, phenobench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "PhenoBench-v110.zip"
    monkeypatch.setattr(
        phenobench.httpx, "stream", lambda *_args, **_kwargs: _FakeStream(b"not-the-phenobench-archive")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        phenobench._download_zip(zip_path)
    assert not zip_path.exists()
    assert not zip_path.with_suffix(zip_path.suffix + ".part").exists()


def test_download_zip_leaves_no_final_archive_when_stream_fails(
    tmp_path: Path, phenobench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "PhenoBench-v110.zip"
    monkeypatch.setattr(
        phenobench.httpx,
        "stream",
        lambda *_args, **_kwargs: _FakeStream(b"partial", error=phenobench.httpx.HTTPError("connection dropped")),
    )
    with pytest.raises(phenobench.httpx.HTTPError, match="connection dropped"):
        phenobench._download_zip(zip_path)
    assert not zip_path.exists()
    assert zip_path.with_suffix(zip_path.suffix + ".part").exists()


def test_main_full_flag_keeps_every_image_in_each_split(
    tmp_path: Path, phenobench: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_prepare(output_dir: Path, *, zip_path: Path, splits: list[str], max_samples: int, size: int) -> Path:
        seen.update(
            {
                "output_dir": output_dir,
                "zip_path": zip_path,
                "splits": splits,
                "max_samples": max_samples,
                "size": size,
            }
        )
        return output_dir / "manifest.csv"

    monkeypatch.setattr(phenobench, "prepare_phenobench", fake_prepare)
    exit_code = phenobench.main(["--full", "--output", str(tmp_path / "out")])
    assert exit_code == 0
    assert seen["max_samples"] == max(phenobench._N_SOURCE_IMAGES.values())
    assert seen["splits"] == ["train", "val"]
