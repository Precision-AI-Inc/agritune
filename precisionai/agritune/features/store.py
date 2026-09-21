# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Persistent storage for previously computed :class:`EncoderFeatures`, one entry per sample.

``DirectoryFeatureStore`` is the development-scale store (one safetensors file + one JSON sidecar
per sample). ``ShardedFeatureStore`` is the production-scale store: many samples' tensors packed
into one safetensors shard file, avoiding one file per image at scale (see
``docs/feature-caching.md``). Both persist enough metadata (shapes, encoder identity, checksum)
per entry to support ``agritune features inspect``/``verify`` without loading full tensors.
"""

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from precisionai.agritune.schemas.features import EncoderFeatures


@dataclass
class FeatureSummary:
    """Lightweight, tensor-free metadata about one stored entry.

    Attributes
    ----------
    key : str
        The cache key this entry is stored under.
    encoder_model : str
    encoder_revision : str | None
    patch_dim : int
    cls_dim : int | None
    num_patches : int
    image_size : tuple[int, int]
        ``(height, width)`` of the single sample this entry represents.
    checksum : str
        SHA-256 hex digest of the underlying tensor file (or shard) at write time.
    """

    key: str
    encoder_model: str
    encoder_revision: str | None
    patch_dim: int
    cls_dim: int | None
    num_patches: int
    image_size: tuple[int, int]
    checksum: str


def _require_single_sample(features: EncoderFeatures) -> None:
    if features.batch_size != 1:
        raise ValueError(f"feature stores hold one sample per entry; got batch_size={features.batch_size}")


def _tensor_dict(features: EncoderFeatures) -> dict[str, torch.Tensor]:
    tensors = {"patch_tokens": features.patch_tokens.contiguous(), "patch_grid": features.patch_grid.contiguous()}
    if features.cls_tokens is not None:
        tensors["cls_tokens"] = features.cls_tokens.contiguous()
    if features.valid_patch_mask is not None:
        tensors["valid_patch_mask"] = features.valid_patch_mask.contiguous()
    return tensors


def _summary_of(key: str, features: EncoderFeatures, *, checksum: str) -> FeatureSummary:
    return FeatureSummary(
        key=key,
        encoder_model=features.encoder_model,
        encoder_revision=features.encoder_revision,
        patch_dim=features.patch_tokens.shape[-1],
        cls_dim=features.cls_tokens.shape[-1] if features.cls_tokens is not None else None,
        num_patches=features.patch_tokens.shape[1],
        image_size=features.image_sizes[0],
        checksum=checksum,
    )


def _features_from_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    image_sizes: list[list[int]],
    encoder_model: str,
    encoder_revision: str | None,
    metadata: dict[str, Any],
) -> EncoderFeatures:
    return EncoderFeatures(
        patch_tokens=tensors["patch_tokens"],
        cls_tokens=tensors.get("cls_tokens"),
        patch_grid=tensors["patch_grid"],
        valid_patch_mask=tensors.get("valid_patch_mask"),
        image_sizes=[(size[0], size[1]) for size in image_sizes],
        encoder_model=encoder_model,
        encoder_revision=encoder_revision,
        metadata=metadata,
    )


class DirectoryFeatureStore:
    """One safetensors file + one JSON sidecar per sample, under a flat directory.

    Intended for development-scale datasets; see :class:`ShardedFeatureStore` for production
    scale.

    Parameters
    ----------
    root : str | Path
        Directory to store entries in; created if missing.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def has(self, key: str) -> bool:
        """Return whether an entry exists for ``key``."""
        return self._tensor_path(key).is_file() and self._meta_path(key).is_file()

    def read(self, key: str) -> EncoderFeatures:
        """Read the full :class:`EncoderFeatures` stored under ``key``."""
        if not self.has(key):
            raise KeyError(key)
        tensors = load_file(str(self._tensor_path(key)))
        meta = json.loads(self._meta_path(key).read_text())
        return _features_from_tensors(
            tensors,
            image_sizes=meta["image_sizes"],
            encoder_model=meta["encoder_model"],
            encoder_revision=meta["encoder_revision"],
            metadata=meta["metadata"],
        )

    def read_summary(self, key: str) -> FeatureSummary:
        """Read only the lightweight, tensor-free :class:`FeatureSummary` for ``key``."""
        if not self.has(key):
            raise KeyError(key)
        meta = json.loads(self._meta_path(key).read_text())
        return FeatureSummary(**meta["summary"])

    def write(self, key: str, features: EncoderFeatures) -> None:
        """Durably persist ``features`` (a single sample, ``batch_size == 1``) under ``key``."""
        _require_single_sample(features)
        tensors = _tensor_dict(features)
        save_file(tensors, str(self._tensor_path(key)))
        checksum = hashlib.sha256(self._tensor_path(key).read_bytes()).hexdigest()
        summary = _summary_of(key, features, checksum=checksum)
        meta = {
            "image_sizes": features.image_sizes,
            "encoder_model": features.encoder_model,
            "encoder_revision": features.encoder_revision,
            "metadata": features.metadata,
            "summary": asdict(summary),
        }
        self._meta_path(key).write_text(json.dumps(meta))

    def flush(self) -> None:
        """No-op — every :meth:`write` is already durable."""

    def list_keys(self) -> list[str]:
        """Return every key currently stored, in no particular order."""
        return [path.stem for path in self._root.glob("*.safetensors")]

    def verify(self, key: str) -> bool:
        """Return whether ``key``'s stored checksum matches its tensor file's actual contents."""
        if not self.has(key):
            raise KeyError(key)
        summary = self.read_summary(key)
        actual = hashlib.sha256(self._tensor_path(key).read_bytes()).hexdigest()
        return actual == summary.checksum

    def clean(self) -> list[str]:
        """Remove any tensor/meta file whose pair is missing (e.g. from an interrupted write).

        Returns
        -------
        list[str]
            Filenames removed.
        """
        tensor_stems = {path.stem for path in self._root.glob("*.safetensors")}
        meta_stems = {path.stem for path in self._root.glob("*.json")}
        removed = []
        for stem in sorted(tensor_stems - meta_stems):
            self._tensor_path(stem).unlink()
            removed.append(f"{stem}.safetensors")
        for stem in sorted(meta_stems - tensor_stems):
            self._meta_path(stem).unlink()
            removed.append(f"{stem}.json")
        return removed

    def _tensor_path(self, key: str) -> Path:
        return self._root / f"{key}.safetensors"

    def _meta_path(self, key: str) -> Path:
        return self._root / f"{key}.json"


class ShardedFeatureStore:
    """Packs many samples' tensors into shard files, indexed by a single JSON manifest.

    Writes are buffered in memory and flushed into a new shard file once ``entries_per_shard``
    samples have accumulated, or when :meth:`flush` is called explicitly (e.g. at the end of a
    build, for the trailing partial shard) — see ``docs/feature-caching.md``.

    Parameters
    ----------
    root : str | Path
        Directory to store shard files and the index in; created if missing.
    entries_per_shard : int, optional
        Samples packed per shard file.
    """

    _INDEX_FILENAME = "shard_index.json"

    def __init__(self, root: str | Path, *, entries_per_shard: int = 1000) -> None:
        if entries_per_shard <= 0:
            raise ValueError(f"entries_per_shard must be positive; got {entries_per_shard}")
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._entries_per_shard = entries_per_shard
        self._index_path = self._root / self._INDEX_FILENAME
        self._index: dict[str, dict[str, Any]] = (
            json.loads(self._index_path.read_text()) if self._index_path.is_file() else {}
        )
        self._pending: dict[str, EncoderFeatures] = {}
        self._shard_cache: tuple[str, dict[str, torch.Tensor]] | None = None
        self._shard_cache_lock = threading.Lock()

    def has(self, key: str) -> bool:
        """Return whether an entry exists for ``key`` (flushed, or buffered in this process)."""
        return key in self._index or key in self._pending

    def read(self, key: str) -> EncoderFeatures:
        """Read the full :class:`EncoderFeatures` stored under ``key``."""
        if key in self._pending:
            return self._pending[key]
        if key not in self._index:
            raise KeyError(key)
        entry = self._index[key]
        tensors = self._load_shard(entry["shard"])
        prefixed = {name[len(key) + 2 :]: tensor for name, tensor in tensors.items() if name.startswith(f"{key}::")}
        return _features_from_tensors(
            prefixed,
            image_sizes=entry["image_sizes"],
            encoder_model=entry["encoder_model"],
            encoder_revision=entry["encoder_revision"],
            metadata=entry["metadata"],
        )

    def read_summary(self, key: str) -> FeatureSummary:
        """Read only the lightweight, tensor-free :class:`FeatureSummary` for ``key``."""
        if key in self._pending:
            checksum = ""  # not yet flushed/checksummed
            return _summary_of(key, self._pending[key], checksum=checksum)
        if key not in self._index:
            raise KeyError(key)
        return FeatureSummary(**self._index[key]["summary"])

    def write(self, key: str, features: EncoderFeatures) -> None:
        """Buffer ``features`` (a single sample, ``batch_size == 1``); auto-flushes when full."""
        _require_single_sample(features)
        self._pending[key] = features
        if len(self._pending) >= self._entries_per_shard:
            self.flush()

    def flush(self) -> None:
        """Write any buffered entries to a new shard file and update the on-disk index."""
        if not self._pending:
            return

        shard_name = f"shard_{len(self._known_shards()):05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        for key, features in self._pending.items():
            for name, tensor in _tensor_dict(features).items():
                tensors[f"{key}::{name}"] = tensor
        save_file(tensors, str(self._root / shard_name))
        checksum = hashlib.sha256((self._root / shard_name).read_bytes()).hexdigest()

        for key, features in self._pending.items():
            self._index[key] = {
                "shard": shard_name,
                "image_sizes": features.image_sizes,
                "encoder_model": features.encoder_model,
                "encoder_revision": features.encoder_revision,
                "metadata": features.metadata,
                "summary": asdict(_summary_of(key, features, checksum=checksum)),
            }
        self._pending.clear()
        self._index_path.write_text(json.dumps(self._index))

    def list_keys(self) -> list[str]:
        """Return every flushed key currently stored, in no particular order."""
        return list(self._index.keys())

    def verify(self, key: str) -> bool:
        """Return whether ``key``'s stored checksum matches its shard file's actual contents.

        Raises
        ------
        KeyError
            If ``key`` is not present at all.
        ValueError
            If ``key`` is only buffered (not yet flushed) and so has no checksum yet.
        """
        if key in self._pending:
            raise ValueError(f"'{key}' has not been flushed yet — no checksum to verify against")
        if key not in self._index:
            raise KeyError(key)
        entry = self._index[key]
        actual = hashlib.sha256((self._root / entry["shard"]).read_bytes()).hexdigest()
        return actual == entry["summary"]["checksum"]

    def clean(self) -> list[str]:
        """Remove shard files not referenced by the index (e.g. from an interrupted write).

        Returns
        -------
        list[str]
            Filenames removed.
        """
        referenced = self._known_shards()
        removed = []
        for shard_path in sorted(self._root.glob("shard_*.safetensors")):
            if shard_path.name not in referenced:
                shard_path.unlink()
                removed.append(shard_path.name)
        return removed

    def _known_shards(self) -> set[str]:
        return {entry["shard"] for entry in self._index.values()}

    def _load_shard(self, shard_name: str) -> dict[str, torch.Tensor]:
        """Load ``shard_name``'s tensors, reusing the most-recently loaded shard when possible.

        Reads within one batch typically land on runs of keys from the same shard (samples are
        packed into shards in stable order), so caching only the single most-recently loaded shard
        avoids re-reading it from disk for every sample it contains, without holding multiple
        shards' tensors in memory at once.

        ``CachedFeatureProvider`` reads a batch's keys from a thread pool, so the check-and-load
        below is lock-guarded: without it, every thread that misses the cache for a not-yet-loaded
        shard would independently re-read that same (potentially multi-gigabyte) shard file from
        disk at once, rather than one thread loading it while the rest wait and reuse the result.
        """
        with self._shard_cache_lock:
            if self._shard_cache is not None and self._shard_cache[0] == shard_name:
                return self._shard_cache[1]
            tensors = load_file(str(self._root / shard_name))
            self._shard_cache = (shard_name, tensors)
            return tensors

    def __enter__(self) -> "ShardedFeatureStore":
        """Return ``self`` for use as a context manager."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Flush any buffered entries on context exit."""
        self.flush()


FEATURE_STORE_TYPES = ("directory", "sharded")


def build_feature_store(
    store_type: str, path: str | Path, *, entries_per_shard: int = 1000
) -> DirectoryFeatureStore | ShardedFeatureStore:
    """Construct the feature store implementation named by ``store_type``.

    The single place every CLI command and service constructs a feature store from a plain
    directory path, so a store built as ``"sharded"`` can actually be read back consistently
    everywhere — see ``docs/feature-caching.md``.

    Parameters
    ----------
    store_type : str
        ``"directory"`` or ``"sharded"``.
    path : str | Path
        Directory to store entries in (created if missing).
    entries_per_shard : int, optional
        Samples packed per shard file; only consulted when ``store_type == "sharded"``.

    Returns
    -------
    DirectoryFeatureStore | ShardedFeatureStore

    Raises
    ------
    ValueError
        If ``store_type`` is not one of :data:`FEATURE_STORE_TYPES`.
    """
    if store_type == "directory":
        return DirectoryFeatureStore(path)
    if store_type == "sharded":
        return ShardedFeatureStore(path, entries_per_shard=entries_per_shard)
    raise ValueError(f"unsupported store_type: {store_type!r}; expected one of {FEATURE_STORE_TYPES}")
