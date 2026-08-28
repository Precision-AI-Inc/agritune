# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""AgriTune — train and evaluate agricultural segmentation decoders on frozen features from a remote ViT encoder.

Package structure
------------------
- ``precisionai.agritune.schemas``       — core dataclasses/protocols (EncoderFeatures, Sample, ...)
- ``precisionai.agritune.data``          — dataset adapters, manifests, split strategies
- ``precisionai.agritune.augmentations`` — deterministic image and feature-space augmentation
- ``precisionai.agritune.encoder``       — EncoderBackend, EncoderGateway, rate limiting, retries
- ``precisionai.agritune.features``      — FeatureProvider, FeatureStore, cache keys
- ``precisionai.agritune.tasks``         — downstream tasks (segmentation first)
- ``precisionai.agritune.training``      — Trainer, evaluator, checkpointing
- ``precisionai.agritune.optimization``  — optimizer/scheduler registries
- ``precisionai.agritune.tracking``      — experiment tracking backends
- ``precisionai.agritune.logging``       — structured logging setup
- ``precisionai.agritune.services``      — orchestration shared by the CLI and API
- ``precisionai.agritune.cli``           — the ``agritune`` command-line entry point
- ``precisionai.agritune.api``           — thin FastAPI layer over ``services``

See ``agritune_implementation_plan.md`` at the repository root for the phased build-out this
package follows, and ``docs/architecture.md`` for the runtime data flow.
"""
