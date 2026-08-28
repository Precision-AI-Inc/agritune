# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Core interfaces separating AgriTune's subsystems.

These protocols exist so concrete subsystems can be implemented and tested independently, and so
that ``precisionai.agritune.training`` and ``precisionai.agritune.tasks`` can depend only on
:class:`FeatureProvider` — never on :class:`EncoderBackend` directly. See
``docs/architecture.md`` for the full layer diagram this set of protocols encodes.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch

from precisionai.agritune.schemas.features import EncoderFeatures
from precisionai.agritune.schemas.samples import PreparedSample

# Narrowed by concrete implementations (Phase 2 dataset adapters, Phase 4 encoder backends):
# a PIL.Image.Image, numpy.ndarray, torch.Tensor, raw bytes, or a data URI/URL string.
ImageInput = Any


@runtime_checkable
class EncoderBackend(Protocol):
    """A source of :class:`EncoderFeatures` for a batch of images.

    Implementations: ``FakeEncoderBackend`` (no network access, used by every unit test) and
    ``RemoteEncoderBackend`` (wraps the hosted embedding API). Only
    ``precisionai.agritune.encoder.gateway.EncoderGateway`` and
    ``precisionai.agritune.features`` may call this — never ``training`` or ``tasks``.
    """

    async def encode(self, images: Sequence[ImageInput]) -> EncoderFeatures:
        """Encode a batch of images.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Images to encode, in whatever representation the concrete backend accepts.

        Returns
        -------
        EncoderFeatures
            Batched features, one entry per input image, in the same order.
        """
        ...


@runtime_checkable
class FeatureAugmentation(Protocol):
    """A feature-space perturbation applied to a batch of :class:`EncoderFeatures` during training.

    Implementations: ``FeatureAugmentationPipeline``. Kept as a protocol (rather than
    :class:`Trainer` importing the concrete pipeline) so ``precisionai.agritune.training`` depends
    only on this shape, matching every other subsystem boundary in this codebase.
    """

    def apply(self, features: EncoderFeatures, *, generator: torch.Generator | None = None) -> EncoderFeatures:
        """Return a (possibly) perturbed copy of ``features``.

        Parameters
        ----------
        features : EncoderFeatures
            Features to augment.
        generator : torch.Generator | None, optional
            RNG source; ``None`` uses PyTorch's global RNG.

        Returns
        -------
        EncoderFeatures
        """
        ...


@runtime_checkable
class FeatureProvider(Protocol):
    """The only thing the trainer and decoders are allowed to obtain features through.

    Implementations: ``CachedFeatureProvider``, ``OnlineFeatureProvider``, ``HybridFeatureProvider``.
    All three expose the same synchronous interface regardless of whether they read from disk or
    call the network underneath — swapping ``features.provider`` in config must never require a
    training-loop code change.
    """

    def get_features(self, samples: Sequence[PreparedSample]) -> EncoderFeatures:
        """Return features for a batch of prepared samples.

        Parameters
        ----------
        samples : Sequence[PreparedSample]
            Augmented samples to obtain features for.

        Returns
        -------
        EncoderFeatures
            Batched features, one entry per input sample, in the same order.
        """
        ...


@runtime_checkable
class FeatureStore(Protocol):
    """Persistent storage for previously computed :class:`EncoderFeatures`.

    Implementations: ``DirectoryFeatureStore`` (development), ``ShardedFeatureStore`` (production).
    """

    def has(self, key: str) -> bool:
        """Return whether a feature entry exists for ``key``."""
        ...

    def read(self, key: str) -> EncoderFeatures:
        """Read the feature entry stored under ``key``.

        Raises
        ------
        KeyError
            If no entry exists for ``key``.
        """
        ...

    def write(self, key: str, features: EncoderFeatures) -> None:
        """Write (or overwrite) the feature entry stored under ``key``."""
        ...


@runtime_checkable
class Decoder(Protocol):
    """A model that maps :class:`EncoderFeatures` to task-specific logits.

    Concrete decoders (e.g. the linear probe, TokenFPN) are ``torch.nn.Module`` subclasses that
    satisfy this interface; the protocol only fixes the shape of the contract.
    """

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        """Compute logits from encoder features.

        Parameters
        ----------
        features : EncoderFeatures
            Batched encoder output.

        Returns
        -------
        torch.Tensor
            Task-specific logits (e.g. per-pixel class logits for segmentation).
        """
        ...


@runtime_checkable
class Task(Protocol):
    """Binds a :class:`Decoder` to a loss and a batch of targets for one downstream task."""

    def forward(self, features: EncoderFeatures) -> Any:
        """Run the decoder and return task-specific outputs (e.g. logits)."""
        ...

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """Compute the scalar training loss for a batch of outputs and targets."""
        ...


@runtime_checkable
class Metric(Protocol):
    """A stateful, accumulating evaluation metric (e.g. mean IoU)."""

    def update(self, outputs: Any, targets: Any) -> None:
        """Accumulate one batch of outputs/targets into the running metric state."""
        ...

    def compute(self) -> dict[str, float]:
        """Return the current metric value(s) computed over all accumulated batches."""
        ...

    def reset(self) -> None:
        """Clear all accumulated state."""
        ...


@runtime_checkable
class Tracker(Protocol):
    """An experiment-tracking backend (JSONL, TensorBoard, MLflow, W&B, ...)."""

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Record a batch of scalar metrics at a given step."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Record run parameters/hyperparameters."""
        ...

    def log_artifact(self, path: str) -> None:
        """Record a file-based artifact (e.g. a checkpoint or prediction visualization)."""
        ...

    def close(self) -> None:
        """Flush and release any resources held by this tracker."""
        ...
