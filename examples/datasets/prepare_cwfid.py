# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Download CWFID and convert it into an AgriTune segmentation manifest.

CWFID (Crop/Weed Field Image Dataset, Haug & Ostermann, ECCV 2014 Workshops) is a small public
agricultural dataset: RGB field images plus colour-coded crop/weed annotation maps. This script
is the worked example of how to turn a public segmentation dataset into AgriTune's CSV manifest
layout (``sample_id,image_path,mask_path,...``).

The original images are **not** redistributed with AgriTune. They are downloaded from the authors'
GitHub mirror and written under ``--output`` (gitignored). Cite the paper if you use the data:

    Haug, S. and Ostermann, J. (2015). A Crop/Weed Field Image Dataset for the Evaluation of
    Computer Vision Based Precision Agriculture Tasks. ECCV 2014 Workshops.

Class map written into the single-channel masks:

=======  ============  ===============
Index    Class         Source colour
=======  ============  ===============
0        background    black (0, 0, 0)
1        crop          green (0, 255, 0)
2        weed          red (255, 0, 0)
=======  ============  ===============
"""

from __future__ import annotations

import argparse
import csv
import sys
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from tqdm import tqdm

_SOURCE_ROOT = "https://raw.githubusercontent.com/cwfid/dataset/master"
_N_SOURCE_IMAGES = 60
_USER_AGENT = "pai-agritune-examples/0.0 (CWFID prep; research use)"
_BACKGROUND = (0, 0, 0)
_CROP = (0, 255, 0)
_WEED = (255, 0, 0)


def _fetch(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"refusing to fetch a non-HTTPS URL: {url}")
    response = httpx.get(url, headers={"User-Agent": _USER_AGENT}, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def _save_resized_rgb(data: bytes, destination: Path, size: int) -> None:
    with Image.open(BytesIO(data)) as image:
        image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR).save(destination)


def _annotation_to_class_mask(data: bytes, size: int) -> Image.Image:
    with Image.open(BytesIO(data)) as annotation:
        rgb = np.array(annotation.convert("RGB"))
    crop = np.all(rgb == np.array(_CROP, dtype=np.uint8), axis=-1)
    weed = np.all(rgb == np.array(_WEED, dtype=np.uint8), axis=-1)
    unknown = ~(np.all(rgb == np.array(_BACKGROUND, dtype=np.uint8), axis=-1) | crop | weed)
    if np.any(unknown):
        raise ValueError("CWFID annotation contains colours other than black/green/red")
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    labels[crop] = 1
    labels[weed] = 2
    mask = Image.fromarray(labels, mode="L")
    return mask.resize((size, size), Image.Resampling.NEAREST)


def prepare_cwfid(output_dir: Path, *, max_samples: int, size: int) -> Path:
    """Download, convert, and write ``manifest.csv`` under ``output_dir``.

    Parameters
    ----------
    output_dir : Path
        Destination root. Images go to ``images/``, class masks to ``masks/``.
    max_samples : int
        How many of the 60 CWFID frames to keep (1-60).
    size : int
        Square side length the RGB image and mask are resized to.

    Returns
    -------
    Path
        Path to the written manifest CSV.
    """
    if not 1 <= max_samples <= _N_SOURCE_IMAGES:
        raise ValueError(f"max_samples must be in [1, {_N_SOURCE_IMAGES}]; got {max_samples}")
    if size < 1:
        raise ValueError(f"size must be positive; got {size}")

    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for index in tqdm(range(1, max_samples + 1), desc="CWFID", unit="sample"):
        stem = f"{index:03d}"
        sample_id = f"cwfid-{stem}"
        image_rel = f"images/{stem}.png"
        mask_rel = f"masks/{stem}.png"
        image_bytes = _fetch(f"{_SOURCE_ROOT}/images/{stem}_image.png")
        annotation_bytes = _fetch(f"{_SOURCE_ROOT}/annotations/{stem}_annotation.png")
        _save_resized_rgb(image_bytes, image_dir / f"{stem}.png", size)
        _annotation_to_class_mask(annotation_bytes, size).save(mask_dir / f"{stem}.png")
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "field_id": "sugar-beet",
                "source": "cwfid",
                "source_index": str(index),
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "image_path", "mask_path", "field_id", "source", "source_index"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest_path} ({len(rows)} samples, {size}x{size}, 3 classes)")
    return manifest_path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CWFID and write an AgriTune segmentation manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/datasets/cwfid"),
        help="Directory to write images/, masks/, and manifest.csv into.",
    )
    sample_count = parser.add_mutually_exclusive_group()
    sample_count.add_argument("--max-samples", type=int, default=24, help="Frames to download (1-60, default 24).")
    sample_count.add_argument(
        "--full",
        action="store_true",
        help=f"Download the complete CWFID set (all {_N_SOURCE_IMAGES} frames); overrides --max-samples.",
    )
    parser.add_argument("--size", type=int, default=384, help="Square resize applied to image and mask (default 384).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Download CWFID and write an AgriTune manifest."""
    args = _parse_args(argv)
    max_samples = _N_SOURCE_IMAGES if args.full else args.max_samples
    try:
        prepare_cwfid(args.output, max_samples=max_samples, size=args.size)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
