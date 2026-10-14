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
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

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


_T = TypeVar("_T")
_R = TypeVar("_R")


def _bounded_map(fn: Callable[[_T], _R], items: Iterable[_T], *, max_workers: int) -> Generator[_R, None, None]:
    """Yield ``fn(item)`` for every item, in order, with at most ``max_workers`` calls in flight.

    Unlike ``ThreadPoolExecutor.map``, which submits every item up front and keeps every finished
    result alive until it is consumed, this only ever holds ``max_workers`` results at once — which
    matters when each result is a whole multi-gigabyte shard's worth of tensors.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be positive; got {max_workers}")
    executor = ThreadPoolExecutor(max_workers=max_workers)
    in_flight: deque[Future[_R]] = deque()
    try:
        for item in items:
            in_flight.append(executor.submit(fn, item))
            if len(in_flight) >= max_workers:
                yield in_flight.popleft().result()
        while in_flight:
            yield in_flight.popleft().result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _group_shard_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Split a shard's flat ``"<key>::<field>"`` tensor names into ``{key: {field: tensor}}``."""
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        key, _, field_name = name.rpartition("::")
        grouped.setdefault(key, {})[field_name] = tensor
    return grouped


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

    def read_many(self, keys: Sequence[str], *, max_workers: int = 16) -> Iterator[tuple[str, EncoderFeatures]]:
        """Yield ``(key, features)`` for every key in ``keys``, in order, reading files concurrently.

        Parameters
        ----------
        keys : Sequence[str]
            Keys to read. Every one must already exist; this is checked before any tensor file
            is read, so a missing key fails fast instead of partway through a long preload.
        max_workers : int, optional
            Files read concurrently.

        Yields
        ------
        tuple[str, EncoderFeatures]

        Raises
        ------
        KeyError
            If any key in ``keys`` has no entry.
        """
        for key in keys:
            if not self.has(key):
                raise KeyError(key)
        yield from _bounded_map(lambda key: (key, self.read(key)), keys, max_workers=max_workers)

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
        self._shard_cache: tuple[str, dict[str, dict[str, torch.Tensor]]] | None = None
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
        tensors = self._load_shard(entry["shard"]).get(key)
        if tensors is None:
            raise KeyError(key)
        return self._entry_features(key, tensors)

    def read_many(self, keys: Sequence[str], *, max_workers: int = 4) -> Iterator[tuple[str, EncoderFeatures]]:
        """Yield ``(key, features)`` for every key in ``keys``, loading each shard file only once.

        :meth:`read` loads a whole shard to return one entry and keeps only the most recent
        shard in memory, so reading keys that are not grouped by shard reloads a shard for
        almost every key. This groups ``keys`` by shard first, loads each shard once (up to
        ``max_workers`` shards concurrently), and yields every requested entry it holds —
        independent of the order ``keys`` arrive in.

        Parameters
        ----------
        keys : Sequence[str]
            Keys to read. Every one must already exist (flushed or buffered); this is checked
            before any shard is read, so a missing key fails fast.
        max_workers : int, optional
            Shards loaded concurrently. Peak memory is roughly ``max_workers`` whole shards.

        Yields
        ------
        tuple[str, EncoderFeatures]
            Grouped by shard, in the order each shard is first referenced by ``keys``; within a
            shard, in ``keys`` order. Entries still buffered in this process come first.

        Raises
        ------
        KeyError
            If any key in ``keys`` has no entry.
        """
        by_shard: dict[str, list[str]] = {}
        buffered: list[str] = []
        for key in keys:
            if key in self._pending:
                buffered.append(key)
            elif key in self._index:
                by_shard.setdefault(self._index[key]["shard"], []).append(key)
            else:
                raise KeyError(key)
        for key in buffered:
            yield key, self._pending[key]

        def load(group: tuple[str, list[str]]) -> list[tuple[str, EncoderFeatures]]:
            shard_name, shard_keys = group
            grouped = _group_shard_tensors(load_file(str(self._root / shard_name)))
            entries = []
            for key in shard_keys:
                if key not in grouped:
                    raise KeyError(key)
                entries.append((key, self._entry_features(key, grouped[key])))
            return entries

        for entries in _bounded_map(load, by_shard.items(), max_workers=max_workers):
            yield from entries

    def _entry_features(self, key: str, tensors: dict[str, torch.Tensor]) -> EncoderFeatures:
        entry = self._index[key]
        return _features_from_tensors(
            tensors,
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

    def _load_shard(self, shard_name: str) -> dict[str, dict[str, torch.Tensor]]:
        """Load ``shard_name``'s tensors grouped by key, reusing the most-recently loaded shard.

        Grouped once per load (``{key: {field: tensor}}``) so each :meth:`read` is a dictionary
        lookup rather than a scan over every tensor name in the shard.

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
            grouped = _group_shard_tensors(load_file(str(self._root / shard_name)))
            self._shard_cache = (shard_name, grouped)
            return grouped

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
