# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Real CLI command handlers, wiring argparse namespaces into ``precisionai.agritune.services``.

Kept separate from ``cli.main`` so parser construction and orchestration logic don't tangle.
"""

import argparse
import asyncio
import json
import sys

from PIL import Image

from precisionai.agritune.cli.config import load_training_run_config
from precisionai.agritune.features.integrity import verify_store
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.manifest import FeatureManifest
from precisionai.agritune.features.precompute import PrecomputeStats
from precisionai.agritune.features.store import DirectoryFeatureStore
from precisionai.agritune.logging import get_logger
from precisionai.agritune.schemas.protocols import EncoderBackend
from precisionai.agritune.services.benchmark_service import run_benchmark
from precisionai.agritune.services.dataset_service import inspect_dataset, validate_dataset
from precisionai.agritune.services.encoder_selection import build_encoder, build_raw_encoder
from precisionai.agritune.services.evaluation_service import EvaluationRunConfig, run_evaluation
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.prediction_service import PredictionRunConfig, run_prediction
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


def _build_raw_encoder(args: argparse.Namespace) -> tuple[EncoderBackend, EncoderFingerprint]:
    return build_raw_encoder(
        base_url=args.base_url, api_key=args.api_key, model=args.model, preprocessing=args.preprocessing
    )


def _build_encoder(args: argparse.Namespace) -> tuple[EncoderBackend, EncoderFingerprint]:
    return build_encoder(
        base_url=args.base_url, api_key=args.api_key, model=args.model, preprocessing=args.preprocessing
    )


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


def features_verify(args: argparse.Namespace) -> int:
    """Handle ``agritune features verify``."""
    store = DirectoryFeatureStore(args.store)
    report = verify_store(store)
    if report.is_valid:
        print(f"OK: {report.total} entries verified.")
        return 0
    print(f"CORRUPTED: {len(report.corrupted_keys)} of {report.total} entries", file=sys.stderr)
    for key in report.corrupted_keys:
        print(f"  {key}", file=sys.stderr)
    return 1


def features_inspect(args: argparse.Namespace) -> int:
    """Handle ``agritune features inspect``."""
    store = DirectoryFeatureStore(args.store)
    manifest = FeatureManifest.from_store(store)
    result = {
        "total_entries": len(manifest),
        "encoder_models": dict(manifest.encoder_models()),
        "patch_dims": dict(manifest.patch_dims()),
    }
    print(json.dumps(result, indent=2))
    return 0


def features_clean(args: argparse.Namespace) -> int:
    """Handle ``agritune features clean``."""
    store = DirectoryFeatureStore(args.store)
    removed = store.clean()
    print(f"removed {len(removed)} file(s)")
    for name in removed:
        print(f"  {name}")
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


def evaluate(args: argparse.Namespace) -> int:
    """Handle ``agritune evaluate``."""
    store = DirectoryFeatureStore(args.store)
    config = EvaluationRunConfig(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        encoder_fingerprint=EncoderFingerprint(
            model=args.encoder_model, revision=args.encoder_revision or None, preprocessing=args.preprocessing
        ),
        decoder_name=args.decoder,
        batch_size=args.batch_size,
        sample_ids=args.sample_ids or None,
    )
    metrics = run_evaluation(config, store=store)
    print(json.dumps(metrics, indent=2))
    return 0


def predict(args: argparse.Namespace) -> int:
    """Handle ``agritune predict``."""
    store = DirectoryFeatureStore(args.store)
    config = PredictionRunConfig(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_classes=args.num_classes,
        encoder_fingerprint=EncoderFingerprint(
            model=args.encoder_model, revision=args.encoder_revision or None, preprocessing=args.preprocessing
        ),
        decoder_name=args.decoder,
        batch_size=args.batch_size,
        sample_ids=args.sample_ids or None,
        write_overlays=args.overlays,
        overlay_alpha=args.overlay_alpha,
    )
    written = run_prediction(config, store=store)
    print(f"wrote {len(written)} prediction(s) to {args.output}")
    return 0


def encoder_benchmark(args: argparse.Namespace) -> int:
    """Handle ``agritune encoder benchmark``."""
    backend, _ = _build_raw_encoder(args)

    def image_factory() -> Image.Image:
        return Image.new("RGB", (args.image_size, args.image_size))

    report = asyncio.run(
        run_benchmark(
            backend,
            batch_sizes=args.batch_sizes,
            concurrencies=args.concurrencies,
            image_factory=image_factory,
            num_requests_per_combination=args.num_requests,
        )
    )

    for result in report.results:
        print(
            f"batch={result.batch_size:>3} concurrency={result.concurrency:>3}  "
            f"{result.images_per_second:6.2f} img/s  "
            f"p50={result.latency_p50_seconds * 1000:6.1f}ms  "
            f"p95={result.latency_p95_seconds * 1000:6.1f}ms  "
            f"errors={result.error_count}"
        )

    best = report.best
    if best is None:
        print("\nNo error-free combination found.", file=sys.stderr)
        return 1

    print("\nRecommended settings")
    print(f"  batch size: {best.batch_size}")
    print(f"  max concurrency: {best.concurrency}")
    print(f"  observed throughput: {best.images_per_second:.2f} images/sec")
    print(f"  p95 request latency: {best.latency_p95_seconds:.2f} sec")
    return 0
