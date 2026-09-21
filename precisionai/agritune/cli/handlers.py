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
from tqdm import tqdm

from precisionai.agritune.augmentations.image.pipeline import AugmentationPipelineConfig, ImageAugmentationPipeline
from precisionai.agritune.cli.config import load_augmentation_selection, load_training_run_config
from precisionai.agritune.features.integrity import verify_store
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.features.manifest import FeatureManifest
from precisionai.agritune.features.migrate import MigrationReport, migrate_store
from precisionai.agritune.features.precompute import PrecomputeStats
from precisionai.agritune.features.store import build_feature_store
from precisionai.agritune.logging import get_logger, redact_text
from precisionai.agritune.schemas.protocols import EncoderBackend
from precisionai.agritune.services.benchmark_service import run_benchmark
from precisionai.agritune.services.config_template_service import write_config_template
from precisionai.agritune.services.dataset_service import (
    inspect_dataset,
    validate_dataset,
    write_manifest_template,
)
from precisionai.agritune.services.encoder_selection import build_encoder, build_raw_encoder
from precisionai.agritune.services.evaluation_service import EvaluationRunConfig, run_evaluation
from precisionai.agritune.services.feature_service import build_features
from precisionai.agritune.services.prediction_service import PredictionRunConfig, run_prediction
from precisionai.agritune.services.training_service import run_training
from precisionai.agritune.tasks.segmentation.losses import SegmentationLossConfig

logger = get_logger(__name__)


def config_init(args: argparse.Namespace) -> int:
    """Handle ``agritune config init``."""
    logger.debug("dispatching config init: output=%s force=%s", args.output, args.force)
    try:
        path = write_config_template(args.output, force=args.force)
    except FileExistsError as err:
        print(str(err), file=sys.stderr)
        return 1
    print(f"wrote config template: {path}")
    return 0


def dataset_validate(args: argparse.Namespace) -> int:
    """Handle ``agritune dataset validate``."""
    logger.debug("dispatching dataset validate: manifest=%s num_classes=%s", args.manifest, args.num_classes)
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
    logger.debug("dispatching dataset inspect: manifest=%s", args.manifest)
    print(json.dumps(inspect_dataset(args.manifest), indent=2))
    return 0


def dataset_init(args: argparse.Namespace) -> int:
    """Handle ``agritune dataset init``."""
    logger.debug("dispatching dataset init: output=%s force=%s", args.output, args.force)
    try:
        path = write_manifest_template(args.output, force=args.force)
    except FileExistsError as err:
        print(str(err), file=sys.stderr)
        return 1
    print(f"wrote manifest template: {path}")
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
    logger.debug("dispatching features build: manifest=%s store=%s", args.manifest, args.store)
    store = build_feature_store(args.store_type, args.store, entries_per_shard=args.entries_per_shard)
    encoder, fingerprint = _build_encoder(args)
    augmentation = load_augmentation_selection(args.augmentation_config)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=augmentation.geometric, photometric=augmentation.photometric)
    )

    bar = tqdm(desc="features build", unit="sample", file=sys.stderr)

    def on_progress(stats: PrecomputeStats) -> None:
        if bar.total is None:
            bar.reset(total=stats.total)
        bar.set_postfix(computed=stats.computed, skipped=stats.skipped, failed=stats.failed)
        bar.update(1)

    try:
        stats = asyncio.run(
            build_features(
                args.manifest,
                store=store,
                encoder=encoder,
                encoder_fingerprint=fingerprint,
                augmentation_mode=augmentation.mode,
                augmentation_pipeline=pipeline,
                global_seed=args.seed,
                augmentation_variant=augmentation.variant,
                on_progress=on_progress,
            )
        )
    finally:
        bar.close()
    print(f"computed={stats.computed} skipped={stats.skipped} failed={stats.failed}")
    if stats.failed:
        print(f"failed sample_ids: {stats.failed_sample_ids}", file=sys.stderr)
        return 1
    return 0


def features_verify(args: argparse.Namespace) -> int:
    """Handle ``agritune features verify``."""
    logger.debug("dispatching features verify: store=%s", args.store)
    store = build_feature_store(args.store_type, args.store, entries_per_shard=args.entries_per_shard)
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
    logger.debug("dispatching features inspect: store=%s", args.store)
    store = build_feature_store(args.store_type, args.store, entries_per_shard=args.entries_per_shard)
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
    logger.debug("dispatching features clean: store=%s", args.store)
    store = build_feature_store(args.store_type, args.store, entries_per_shard=args.entries_per_shard)
    removed = store.clean()
    print(f"removed {len(removed)} file(s)")
    for name in removed:
        print(f"  {name}")
    return 0


def features_migrate(args: argparse.Namespace) -> int:
    """Handle ``agritune features migrate``."""
    logger.debug("dispatching features migrate: source=%s dest=%s", args.source, args.dest)
    source = build_feature_store(args.source_type, args.source)
    dest = build_feature_store(args.dest_type, args.dest, entries_per_shard=args.dest_entries_per_shard)

    bar = tqdm(desc="migrating features", unit="sample", file=sys.stderr)

    def on_progress(report: MigrationReport) -> None:
        if bar.total is None:
            bar.reset(total=report.total)
        bar.set_postfix(migrated=report.migrated, skipped=report.skipped)
        bar.update(1)

    try:
        report = migrate_store(source, dest, on_progress=on_progress)
    finally:
        bar.close()
    print(f"migrated={report.migrated} skipped={report.skipped} total={report.total}")
    return 0


def train(args: argparse.Namespace) -> int:
    """Handle ``agritune train``."""
    logger.debug("dispatching train: config=%s overrides=%s", args.config, redact_text(repr(args.overrides)))
    config = load_training_run_config(args.config, args.overrides)
    store = build_feature_store(config.store_type, config.feature_store_dir, entries_per_shard=config.entries_per_shard)
    result = run_training(config, store=store, show_progress=True)
    print(f"run directory: {result.run_directory.path}")
    print(f"final epoch: {result.final_train_state['epoch']}")
    print(f"train metrics: {result.train_metrics}")
    print(f"val metrics: {result.val_metrics}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    """Handle ``agritune evaluate``."""
    logger.debug("dispatching evaluate: manifest=%s checkpoint=%s", args.manifest, args.checkpoint)
    store = build_feature_store(args.store_type, args.store, entries_per_shard=args.entries_per_shard)
    augmentation = load_augmentation_selection(args.augmentation_config)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=augmentation.geometric, photometric=augmentation.photometric)
    )
    config = EvaluationRunConfig(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        encoder_fingerprint=EncoderFingerprint(
            model=args.encoder_model, revision=args.encoder_revision or None, preprocessing=args.preprocessing
        ),
        decoder_name=args.decoder,
        decoder_kwargs=json.loads(args.decoder_kwargs),
        batch_size=args.batch_size,
        sample_ids=args.sample_ids or None,
        device=args.device,
        loss=SegmentationLossConfig(
            name=args.loss_name,
            ignore_index=args.loss_ignore_index,
            ce_weight=args.loss_ce_weight,
            dice_weight=args.loss_dice_weight,
        ),
        resize=tuple(args.resize) if args.resize else None,
        augmentation_mode=augmentation.mode,
        augmentation_pipeline=pipeline,
        global_seed=args.seed,
        augmentation_variant=augmentation.variant,
    )
    metrics = run_evaluation(config, store=store, show_progress=True)
    print(json.dumps(metrics, indent=2))
    return 0


def predict(args: argparse.Namespace) -> int:
    """Handle ``agritune predict``."""
    logger.debug("dispatching predict: manifest=%s checkpoint=%s", args.manifest, args.checkpoint)
    store = build_feature_store(args.store_type, args.store, entries_per_shard=args.entries_per_shard)
    augmentation = load_augmentation_selection(args.augmentation_config)
    pipeline = ImageAugmentationPipeline(
        AugmentationPipelineConfig(geometric=augmentation.geometric, photometric=augmentation.photometric)
    )
    config = PredictionRunConfig(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_classes=args.num_classes,
        encoder_fingerprint=EncoderFingerprint(
            model=args.encoder_model, revision=args.encoder_revision or None, preprocessing=args.preprocessing
        ),
        decoder_name=args.decoder,
        decoder_kwargs=json.loads(args.decoder_kwargs),
        batch_size=args.batch_size,
        sample_ids=args.sample_ids or None,
        device=args.device,
        resize=tuple(args.resize) if args.resize else None,
        write_overlays=args.overlays,
        overlay_alpha=args.overlay_alpha,
        augmentation_mode=augmentation.mode,
        augmentation_pipeline=pipeline,
        global_seed=args.seed,
        augmentation_variant=augmentation.variant,
    )
    written = run_prediction(config, store=store, show_progress=True)
    print(f"wrote {len(written)} prediction(s) to {args.output}")
    return 0


def encoder_benchmark(args: argparse.Namespace) -> int:
    """Handle ``agritune encoder benchmark``."""
    logger.debug("dispatching encoder benchmark: base_url=%s model=%s", args.base_url, args.model)
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
            show_progress=True,
        )
    )

    for result in report.results:
        line = (
            f"batch={result.batch_size:>3} concurrency={result.concurrency:>3}  "
            f"{result.images_per_second:6.2f} img/s  "
            f"p50={result.latency_p50_seconds * 1000:6.1f}ms  "
            f"p95={result.latency_p95_seconds * 1000:6.1f}ms  "
            f"errors={result.error_count}"
        )
        if result.sample_error is not None:
            line += f"  ({result.sample_error})"
        print(line)

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
