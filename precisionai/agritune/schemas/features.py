# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The canonical feature representation returned by any :class:`EncoderBackend`.

Nothing here may assume a fixed patch dimension, CLS dimension, patch grid shape, or square
image — the real encoder is served behind an API where all four vary by model and, with
``native_resolution`` enabled, by image (see ``docs/encoder.md``).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch


def _validate_token_values(patch_tokens: torch.Tensor, cls_tokens: torch.Tensor | None, *, batch_size: int) -> None:
    if not patch_tokens.is_floating_point():
        raise ValueError(f"patch_tokens must use a floating-point dtype; got {patch_tokens.dtype}")
    if not torch.all(torch.isfinite(patch_tokens)):
        raise ValueError("patch_tokens must contain only finite values")
    if cls_tokens is None:
        return
    if cls_tokens.ndim != 2 or cls_tokens.shape[0] != batch_size or cls_tokens.shape[1] < 1:
        raise ValueError(
            f"cls_tokens must have shape (B, D_cls) with B={batch_size} and D_cls > 0; got {tuple(cls_tokens.shape)}"
        )
    if not cls_tokens.is_floating_point():
        raise ValueError(f"cls_tokens must use a floating-point dtype; got {cls_tokens.dtype}")
    if not torch.all(torch.isfinite(cls_tokens)):
        raise ValueError("cls_tokens must contain only finite values")


def _validate_image_sizes(image_sizes: list[tuple[int, int]], *, batch_size: int) -> None:
    if len(image_sizes) != batch_size:
        raise ValueError(f"image_sizes must have length B={batch_size}; got {len(image_sizes)}")
    for index, image_size in enumerate(image_sizes):
        if (
            len(image_size) != 2
            or any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in image_size)
            or any(dimension <= 0 for dimension in image_size)
        ):
            raise ValueError(f"image_sizes[{index}] must contain two positive integers; got {image_size}")


def _validate_patch_grid(patch_grid: torch.Tensor, *, batch_size: int) -> None:
    if patch_grid.shape != (batch_size, 2):
        raise ValueError(f"patch_grid must have shape (B, 2) with B={batch_size}; got {tuple(patch_grid.shape)}")
    if patch_grid.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"patch_grid must use an integer dtype; got {patch_grid.dtype}")
    if torch.any(patch_grid <= 0):
        raise ValueError(f"patch_grid dimensions must be positive; got {patch_grid.tolist()}")


def _validate_valid_patch_mask(valid_patch_mask: torch.Tensor | None, *, batch_size: int, num_patches: int) -> None:
    if valid_patch_mask is None:
        return
    if valid_patch_mask.shape != (batch_size, num_patches):
        raise ValueError(
            f"valid_patch_mask must have shape (B, N)=({batch_size}, {num_patches}); got "
            f"{tuple(valid_patch_mask.shape)}"
        )
    if valid_patch_mask.dtype != torch.bool:
        raise ValueError(f"valid_patch_mask must use dtype torch.bool; got {valid_patch_mask.dtype}")


@dataclass
class EncoderFeatures:
    """Batched output of an :class:`EncoderBackend`, ready for a downstream :class:`Task`.

    Samples in a batch may have different patch grids (e.g. under native-resolution encoding);
    ``patch_tokens`` is padded to the largest patch count in the batch, and ``valid_patch_mask``
    marks which entries are real versus padding.

    Attributes
    ----------
    patch_tokens : torch.Tensor
        Shape ``(B, N, D)`` where ``N`` is the padded patch count for the batch and ``D`` is the
        patch embedding dimension. Padding positions (where ``valid_patch_mask`` is ``False``)
        hold unspecified values and must not be read by a decoder.
    cls_tokens : torch.Tensor | None
        Shape ``(B, D_cls)``, or ``None`` if the encoder/model does not produce a CLS token.
        ``D_cls`` is independent of ``D`` — never assume they are equal.
    patch_grid : torch.Tensor
        Integer tensor of shape ``(B, 2)``, each row ``(H, W)`` giving the patch grid for that
        sample. ``H * W`` need not be square and need not match another sample in the batch.
    valid_patch_mask : torch.Tensor | None
        Boolean tensor of shape ``(B, N)``, ``True`` where ``patch_tokens`` holds a real patch.
        ``None`` is shorthand for "every sample fills all N patches" (no padding in this batch).
    image_sizes : list[tuple[int, int]]
        Original ``(height, width)`` in pixels for each sample, length ``B``.
    encoder_model : str
        The model identifier/alias used to produce these features (e.g. ``"pai-embedding"``).
    encoder_revision : str | None
        Encoder revision, when the backend can supply one. The hosted encoder API exposes no
        queryable revision, so this is commonly ``None`` for :class:`RemoteEncoderBackend` — see
        ``docs/encoder.md``.
    metadata : dict[str, Any]
        Free-form additional metadata (e.g. preprocessing parameters, dtype notes). Never required
        by downstream code.
    """

    patch_tokens: torch.Tensor
    cls_tokens: torch.Tensor | None
    patch_grid: torch.Tensor
    valid_patch_mask: torch.Tensor | None
    image_sizes: list[tuple[int, int]]
    encoder_model: str
    encoder_revision: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate internal shape consistency.

        Raises
        ------
        ValueError
            If any tensor's batch dimension, patch count, or grid is inconsistent with the others.
        """
        if self.patch_tokens.ndim != 3:
            raise ValueError(f"patch_tokens must have shape (B, N, D); got {tuple(self.patch_tokens.shape)}")
        batch_size, num_patches, patch_dim = self.patch_tokens.shape
        if batch_size < 1 or num_patches < 1 or patch_dim < 1:
            raise ValueError(
                f"patch_tokens dimensions B, N, and D must be positive; got {tuple(self.patch_tokens.shape)}"
            )
        _validate_token_values(self.patch_tokens, self.cls_tokens, batch_size=batch_size)
        _validate_patch_grid(self.patch_grid, batch_size=batch_size)
        _validate_valid_patch_mask(self.valid_patch_mask, batch_size=batch_size, num_patches=num_patches)

        _validate_image_sizes(self.image_sizes, batch_size=batch_size)

        grid_products = (self.patch_grid[:, 0] * self.patch_grid[:, 1]).tolist()
        for index, expected_patches in enumerate(grid_products):
            valid_count = (
                int(self.valid_patch_mask[index].sum().item()) if self.valid_patch_mask is not None else num_patches
            )
            if expected_patches != valid_count:
                raise ValueError(
                    f"sample {index}: patch_grid implies {expected_patches} patches but {valid_count} are marked valid"
                )

    @property
    def batch_size(self) -> int:
        """Return the batch size ``B``."""
        return self.patch_tokens.shape[0]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "EncoderFeatures":
        """Return a copy with every tensor field moved to ``device``.

        ``patch_tokens``, ``cls_tokens``, and ``valid_patch_mask`` are the only fields a decoder
        ever computes with, so a decoder on ``device`` needs all three there too; ``patch_grid`` is
        moved as well purely for consistency (nothing prevents a decoder from indexing into it
        directly). ``image_sizes``/``encoder_model``/``encoder_revision``/``metadata`` are plain
        Python values, unaffected.

        Parameters
        ----------
        device : torch.device | str
            Target device (e.g. ``torch.device("cuda:3")`` or ``"cuda:3"``).
        non_blocking : bool, optional
            Passed through to every tensor's own ``.to()`` — only actually asynchronous when the
            source tensor is in pinned host memory and the target is a CUDA device.

        Returns
        -------
        EncoderFeatures
        """
        return replace(
            self,
            patch_tokens=self.patch_tokens.to(device, non_blocking=non_blocking),
            cls_tokens=self.cls_tokens.to(device, non_blocking=non_blocking) if self.cls_tokens is not None else None,
            patch_grid=self.patch_grid.to(device, non_blocking=non_blocking),
            valid_patch_mask=(
                self.valid_patch_mask.to(device, non_blocking=non_blocking)
                if self.valid_patch_mask is not None
                else None
            ),
        )

    def patch_grid_hw(self, index: int) -> tuple[int, int]:
        """Return the ``(H, W)`` patch grid for one sample in the batch.

        Parameters
        ----------
        index : int
            Position of the sample within the batch.

        Returns
        -------
        tuple[int, int]
            ``(H, W)`` patch grid for that sample.
        """
        height, width = self.patch_grid[index].tolist()
        return int(height), int(width)

    def uniform_patch_grid(self) -> tuple[int, int]:
        """Return the ``(H, W)`` patch grid, requiring every sample in the batch to share it.

        Most decoders reshape ``patch_tokens`` into a single ``(B, H, W, D)`` spatial grid, which
        requires a uniform grid across the batch — the normal case when a fixed input resolution
        is enforced upstream (e.g. via ``augmentation.geometric.resize``). Native-resolution
        batches with differing grids must be handled per-sample instead.

        Returns
        -------
        tuple[int, int]
            The shared ``(H, W)`` patch grid.

        Raises
        ------
        ValueError
            If samples in the batch have different patch grids.
        """
        first = self.patch_grid_hw(0)
        for index in range(1, self.batch_size):
            other = self.patch_grid_hw(index)
            if other != first:
                raise ValueError(
                    f"batch has non-uniform patch grids (sample 0: {first}, sample {index}: {other}) "
                    "— this operation requires every sample in the batch to share the same patch grid"
                )
        return first


def concatenate_encoder_features(features_list: Sequence[EncoderFeatures]) -> EncoderFeatures:
    """Concatenate multiple :class:`EncoderFeatures` along the batch dimension.

    Used by :class:`~precisionai.agritune.encoder.gateway.EncoderGateway` to merge per-batch
    encoder responses back into one result spanning the original request. Patch tokens are
    re-padded to the largest patch count across every input.

    Parameters
    ----------
    features_list : Sequence[EncoderFeatures]
        Non-empty sequence of features to concatenate, in order. Every entry must share
        ``encoder_model``/``encoder_revision``, patch dimension, and either all have ``cls_tokens``
        or all have none.

    Returns
    -------
    EncoderFeatures
        A single batch containing every sample from ``features_list``, in order.

    Raises
    ------
    ValueError
        If ``features_list`` is empty, or entries disagree on encoder identity, patch dimension,
        or CLS-token presence.
    """
    if not features_list:
        raise ValueError("features_list must be non-empty")

    first = features_list[0]
    patch_dim = first.patch_tokens.shape[2]
    has_cls = first.cls_tokens is not None
    _validate_concatenation_inputs(features_list, patch_dim=patch_dim, has_cls=has_cls, first=first)

    max_patches = max(entry.patch_tokens.shape[1] for entry in features_list)
    total_batch = sum(entry.batch_size for entry in features_list)

    padded = torch.zeros(total_batch, max_patches, patch_dim, dtype=first.patch_tokens.dtype)
    valid_mask = torch.zeros(total_batch, max_patches, dtype=torch.bool)

    offset = 0
    grids = []
    image_sizes: list[tuple[int, int]] = []
    for entry in features_list:
        batch_size, num_patches, _ = entry.patch_tokens.shape
        padded[offset : offset + batch_size, :num_patches] = entry.patch_tokens
        if entry.valid_patch_mask is not None:
            valid_mask[offset : offset + batch_size, :num_patches] = entry.valid_patch_mask
        else:
            valid_mask[offset : offset + batch_size, :num_patches] = True
        offset += batch_size
        grids.append(entry.patch_grid)
        image_sizes.extend(entry.image_sizes)

    needs_mask = max_patches != min(entry.patch_tokens.shape[1] for entry in features_list) or any(
        entry.valid_patch_mask is not None for entry in features_list
    )

    cls_tokens: torch.Tensor | None = None
    if has_cls:
        cls_list: list[torch.Tensor] = []
        for entry in features_list:
            if entry.cls_tokens is None:  # pragma: no cover — excluded by the has_cls check above
                raise RuntimeError("inconsistent cls_tokens presence despite prior validation")
            cls_list.append(entry.cls_tokens)
        cls_tokens = torch.cat(cls_list, dim=0)

    return EncoderFeatures(
        patch_tokens=padded,
        cls_tokens=cls_tokens,
        patch_grid=torch.cat(grids, dim=0),
        valid_patch_mask=valid_mask if needs_mask else None,
        image_sizes=image_sizes,
        encoder_model=first.encoder_model,
        encoder_revision=first.encoder_revision,
    )


def select_one(features: EncoderFeatures, index: int) -> EncoderFeatures:
    """Slice one sample out of a batched :class:`EncoderFeatures`, trimmed to its own patch count.

    Used by :class:`~precisionai.agritune.features.provider.HybridFeatureProvider` to write a
    single sample's features to a :class:`~precisionai.agritune.schemas.protocols.FeatureStore`
    after encoding a whole batch of cache misses in one gateway call. The returned entry has no
    padding: ``valid_patch_mask`` is ``None`` because a length-1 batch has nothing to pad against.

    Parameters
    ----------
    features : EncoderFeatures
        A batch to slice from.
    index : int
        Position of the sample to extract.

    Returns
    -------
    EncoderFeatures
        A length-1 batch containing only that sample.
    """
    if index < 0 or index >= features.batch_size:
        raise ValueError(f"index must be in [0, {features.batch_size}); got {index}")
    height, width = features.patch_grid_hw(index)
    num_patches = height * width
    tokens = features.patch_tokens[index]
    if features.valid_patch_mask is not None:
        tokens = tokens[features.valid_patch_mask[index]]
    else:
        tokens = tokens[:num_patches]
    return EncoderFeatures(
        patch_tokens=tokens.unsqueeze(0),
        cls_tokens=features.cls_tokens[index : index + 1] if features.cls_tokens is not None else None,
        patch_grid=features.patch_grid[index : index + 1],
        valid_patch_mask=None,
        image_sizes=[features.image_sizes[index]],
        encoder_model=features.encoder_model,
        encoder_revision=features.encoder_revision,
        metadata=dict(features.metadata),
    )


def _validate_concatenation_inputs(
    features_list: Sequence[EncoderFeatures], *, patch_dim: int, has_cls: bool, first: EncoderFeatures
) -> None:
    for other in features_list[1:]:
        if other.patch_tokens.shape[2] != patch_dim:
            raise ValueError("all entries must share the same patch dimension")
        if other.patch_tokens.dtype != first.patch_tokens.dtype:
            raise ValueError("all entries must share the same patch-token dtype")
        if (other.cls_tokens is not None) != has_cls:
            raise ValueError("all entries must agree on whether cls_tokens is present")
        if has_cls and other.cls_tokens is not None and first.cls_tokens is not None:
            if other.cls_tokens.shape[1] != first.cls_tokens.shape[1]:
                raise ValueError("all entries must share the same CLS dimension")
            if other.cls_tokens.dtype != first.cls_tokens.dtype:
                raise ValueError("all entries must share the same CLS-token dtype")
        if other.encoder_model != first.encoder_model or other.encoder_revision != first.encoder_revision:
            raise ValueError("all entries must share the same encoder_model/encoder_revision")
