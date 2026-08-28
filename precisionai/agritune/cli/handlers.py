# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Real CLI command handlers, wiring argparse namespaces into ``precisionai.agritune.services``.

Kept separate from ``cli.main`` so parser construction and orchestration logic don't tangle.
"""

import argparse
import asyncio
import json
import sys

from precisionai.agritune.cli.config import load_training_run_config
from precisionai.agritune.encoder.fake import FakeEncoderBackend
from precisionai.agritune.encoder.gateway import EncoderGateway
from precisionai.agritune.encoder.remote import RemoteEncoderBackend, RemoteEncoderConfig
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.precompute import PrecomputeStats
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.protocols import EncoderBackend
from precisionai.agritune.services.dataset_service import inspect_dataset, validate_dataset
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.training_service import run_training

logger = get_logger(__name__)


def dataset_validate(args: argparse.Namespace) -> int:
    """Handle ``agritune dataset validate``."""
    report = validate_dataset(args.manifest, num_classes=args.num_classes, ignore_index=args.ignore_index)
    if report.is_valid:
        print("OK: no issues found.")
        return 0
    for issue in report.issues:
        print(f"[{issue.category}] {issue.sample_id or '-'}: {issue.message}", file=sys.stderr)
    print(f"\n{len(report.issues)} issue(s) found.", file=sys.stderr)
    return 1


def dataset_inspect(args: argparse.Namespace) -> int:
    """Handle ``agritune dataset inspect``."""
    print(json.dumps(inspect_dataset(args.manifest), indent=2))
    return 0


def _build_encoder(args: argparse.Namespace) -> tuple[EncoderBackend, EncoderFingerprint]:
    if args.base_url:
        backend: EncoderBackend = RemoteEncoderBackend(
            RemoteEncoderConfig(base_url=args.base_url, api_key=args.api_key or "", model=args.model)
        )
        fingerprint = EncoderFingerprint(model=args.model, revision=None, preprocessing=args.preprocessing)
    else:
        backend = FakeEncoderBackend()
        fingerprint = EncoderFingerprint(model="fake-encoder", revision="fake-v1", preprocessing=args.preprocessing)
    return EncoderGateway(backend), fingerprint


def features_build(args: argparse.Namespace) -> int:
    """Handle ``agritune features build``."""
    store = DirectoryFeatureStore(args.store)
    encoder, fingerprint = _build_encoder(args)

    def on_progress(stats: PrecomputeStats) -> None:
        done = stats.computed + stats.skipped + stats.failed
        print(
            f"\r{done}/{stats.total} (computed={stats.computed} skipped={stats.skipped} failed={stats.failed})",
            end="",
            file=sys.stderr,
        )

    stats = asyncio.run(
        build_features(
            args.manifest, store=store, encoder=encoder, encoder_fingerprint=fingerprint, on_progress=on_progress
        )
    )
    print(file=sys.stderr)
    print(f"computed={stats.computed} skipped={stats.skipped} failed={stats.failed}")
    if stats.failed:
        print(f"failed sample_ids: {stats.failed_sample_ids}", file=sys.stderr)
        return 1
    return 0


def train(args: argparse.Namespace) -> int:
    """Handle ``agritune train``."""
    config = load_training_run_config(args.config, args.overrides)
    store = DirectoryFeatureStore(config.feature_store_dir)
    result = run_training(config, store=store)
    print(f"run directory: {result.run_directory.path}")
    print(f"final epoch: {result.final_train_state['epoch']}")
    print(f"val metrics: {result.val_metrics}")
    return 0
