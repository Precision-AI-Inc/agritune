# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Download PhenoBench and convert it into an AgriTune segmentation manifest.

PhenoBench (Weyler et al., IEEE TPAMI 2024; arXiv:2306.04557) is a large public agricultural
dataset: 1024x1024 UAV images of real sugar-beet fields across many growth stages, with dense
pixel-wise crop/weed semantic annotations. It is the "bigger, real-world" counterpart to the
60-image CWFID worked example (see ``prepare_cwfid.py``) -- 1,407 training and 772 validation
images with public annotations (the 693-image test split's labels are withheld by the authors
for their leaderboard, so this script only converts ``train``/``val``).

Unlike CWFID, PhenoBench is not published as individually fetchable files: the authors distribute
one combined archive. This script downloads that archive once (~7.6 GB, cached under
``--zip-cache`` so re-runs with a different ``--max-samples``/``--size``/``--splits`` do not
re-download it) and converts a chosen number of samples per split into AgriTune's CSV manifest
layout (``sample_id,image_path,mask_path,...``).

The original images are **not** redistributed with AgriTune. They are downloaded from the
authors' site (https://www.phenobench.org) and written under ``--output`` (gitignored). The
dataset is licensed CC BY-SA 4.0 -- derivatives (including the files this script writes) carry
the same license. Cite the paper if you use the data:

    Weyler, J., Magistri, F., Marks, E., Chong, Y.L., Sodano, M., Roggiolani, G., Chebrolu, N.,
    Stachniss, C. and Behley, J. (2024). PhenoBench: A Large Dataset and Benchmarks for Semantic
    Image Interpretation in the Agricultural Domain. IEEE Transactions on Pattern Analysis and
    Machine Intelligence.

Source semantic label ids, and how this script collapses them into AgriTune's 3-class scheme
(matching CWFID's background/crop/weed convention):

=======  =================  ===============================================
Source   Source meaning     AgriTune class
=======  =================  ===============================================
0        background/soil    0 (background)
1        crop               1 (crop)
2        weed                2 (weed)
3        partial crop        1 (crop)  -- <50% visible, merged into crop
4        partial weed        2 (weed)  -- <50% visible, merged into weed
=======  =================  ===============================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from tqdm import tqdm

_ZIP_URL = "https://www.phenobench.org/data/PhenoBench-v110.zip"
_ZIP_MD5 = "5168bba762053725890478432cdbdb1d"
_ZIP_SIZE_BYTES = 7_630_658_167
_ARCHIVE_ROOT = "PhenoBench"
_N_SOURCE_IMAGES = {"train": 1407, "val": 772}
_USER_AGENT = "pai-agritune-examples/0.0 (PhenoBench prep; research use)"
_LABEL_LUT = np.array([0, 1, 2, 1, 2], dtype=np.uint8)  # source id -> background/crop/weed


def _download_zip(zip_path: Path) -> None:
    if zip_path.exists() and zip_path.stat().st_size == _ZIP_SIZE_BYTES:
        print(f"using cached archive {zip_path} ({_ZIP_SIZE_BYTES / 1e9:.1f} GB)")
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".part")
    digest = hashlib.md5(usedforsecurity=False)
    timeout = httpx.Timeout(30.0, read=300.0, write=300.0)
    with httpx.stream(
        "GET", _ZIP_URL, headers={"User-Agent": _USER_AGENT}, timeout=timeout, follow_redirects=True
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", _ZIP_SIZE_BYTES))
        with (
            tmp_path.open("wb") as handle,
            tqdm(desc="downloading PhenoBench", total=total, unit="B", unit_scale=True, unit_divisor=1024) as bar,
        ):
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                handle.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))

    if digest.hexdigest() != _ZIP_MD5:
        tmp_path.unlink()
        raise ValueError(f"downloaded archive checksum mismatch (expected md5 {_ZIP_MD5}); deleted, re-run to retry")
    tmp_path.replace(zip_path)
    print(f"downloaded and verified {zip_path}")


def _list_split_samples(archive: zipfile.ZipFile, split: str) -> list[str]:
    prefix = f"{_ARCHIVE_ROOT}/{split}/images/"
    return sorted(
        name[len(prefix) :] for name in archive.namelist() if name.startswith(prefix) and name.endswith(".png")
    )


def _save_resized_rgb(data: bytes, destination: Path, size: int) -> None:
    with Image.open(BytesIO(data)) as image:
        image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR).save(destination)


def _semantics_to_class_mask(data: bytes, size: int) -> Image.Image:
    with Image.open(BytesIO(data)) as annotation:
        raw = np.array(annotation)
    if raw.min() < 0 or raw.max() >= len(_LABEL_LUT):
        raise ValueError(f"PhenoBench semantic mask contains a label id outside [0, {len(_LABEL_LUT) - 1}]")
    labels = _LABEL_LUT[raw]
    mask = Image.fromarray(labels, mode="L")
    return mask.resize((size, size), Image.Resampling.NEAREST)


def prepare_phenobench(output_dir: Path, *, zip_path: Path, splits: list[str], max_samples: int, size: int) -> Path:
    """Download (if needed) and convert PhenoBench train/val samples into a manifest.

    Parameters
    ----------
    output_dir : Path
        Destination root. Images go to ``images/``, class masks to ``masks/``.
    zip_path : Path
        Where the ~7.6 GB source archive is cached; reused across runs.
    splits : list[str]
        Which of ``"train"``/``"val"`` to convert. PhenoBench's ``test`` split annotations are
        withheld by the authors for their leaderboard, so it is not supported here.
    max_samples : int
        How many images to keep from *each* requested split.
    size : int
        Square side length the RGB image and mask are resized to.

    Returns
    -------
    Path
        Path to the written manifest CSV.
    """
    if not splits:
        raise ValueError("splits must not be empty")
    invalid_splits = sorted(set(splits) - set(_N_SOURCE_IMAGES))
    if invalid_splits:
        raise ValueError(f"unsupported split(s) {invalid_splits}; choose from {sorted(_N_SOURCE_IMAGES)}")
    if max_samples < 1:
        raise ValueError(f"max_samples must be positive; got {max_samples}")
    if size < 1:
        raise ValueError(f"size must be positive; got {size}")

    _download_zip(zip_path)

    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for split in splits:
            names = _list_split_samples(archive, split)[:max_samples]
            for index, name in tqdm(
                enumerate(names, start=1), total=len(names), desc=f"PhenoBench {split}", unit="sample"
            ):
                stem = Path(name).stem
                sample_id = f"phenobench-{split}-{stem}"
                image_rel = f"images/{split}_{stem}.png"
                mask_rel = f"masks/{split}_{stem}.png"
                image_bytes = archive.read(f"{_ARCHIVE_ROOT}/{split}/images/{name}")
                semantics_bytes = archive.read(f"{_ARCHIVE_ROOT}/{split}/semantics/{name}")
                _save_resized_rgb(image_bytes, image_dir / f"{split}_{stem}.png", size)
                _semantics_to_class_mask(semantics_bytes, size).save(mask_dir / f"{split}_{stem}.png")
                date, _frame_index, plot_id = stem.split("_", 2)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": image_rel,
                        "mask_path": mask_rel,
                        "field_id": plot_id,
                        "capture_date": date,
                        "split": split,
                        "source": "phenobench",
                        "source_index": str(index),
                    }
                )

    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "sample_id",
        "image_path",
        "mask_path",
        "field_id",
        "capture_date",
        "split",
        "source",
        "source_index",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest_path} ({len(rows)} samples, {size}x{size}, 3 classes)")
    return manifest_path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PhenoBench and write an AgriTune segmentation manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/datasets/phenobench"),
        help="Directory to write images/, masks/, and manifest.csv into.",
    )
    parser.add_argument(
        "--zip-cache",
        type=Path,
        default=None,
        help="Where to cache the ~7.6 GB source archive (default: <output>/.cache/PhenoBench-v110.zip).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(_N_SOURCE_IMAGES),
        default=["train", "val"],
        help="PhenoBench splits to convert (default: train val; test annotations are withheld by the authors).",
    )
    sample_count = parser.add_mutually_exclusive_group()
    sample_count.add_argument(
        "--max-samples", type=int, default=40, help="Images to keep from each requested split (default 40)."
    )
    sample_count.add_argument(
        "--full",
        action="store_true",
        help="Keep every image in each requested split (1407 train / 772 val); overrides --max-samples.",
    )
    parser.add_argument("--size", type=int, default=384, help="Square resize applied to image and mask (default 384).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Download PhenoBench and write an AgriTune manifest."""
    args = _parse_args(argv)
    zip_path = args.zip_cache if args.zip_cache is not None else args.output / ".cache" / "PhenoBench-v110.zip"
    max_samples = max(_N_SOURCE_IMAGES.values()) if args.full else args.max_samples
    try:
        prepare_phenobench(args.output, zip_path=zip_path, splits=args.splits, max_samples=max_samples, size=args.size)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
