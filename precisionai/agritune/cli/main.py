# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The ``agritune`` command-line entry point.

Run ``agritune --help`` to see available commands.

Every subcommand's handler calls into ``precisionai.agritune.services`` — the API layer calls the
same services, so logic is never duplicated between entry points.
"""

import argparse
import os
import sys
from collections.abc import Callable, Sequence

from precisionai.agritune.cli import handlers
from precisionai.agritune.logging import configure_logging, get_logger
from precisionai.agritune.utils.env import ENCODER_API_KEY_VARIABLE, load_env_file

logger = get_logger(__name__)

_CommandHandler = Callable[[argparse.Namespace], int]


def _build_dataset_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("dataset", help="Validate and inspect agricultural datasets.")
    dataset_subparsers = parser.add_subparsers(dest="subcommand", required=True)

    validate = dataset_subparsers.add_parser("validate", help="Check a dataset manifest for errors.")
    validate.add_argument("--manifest", required=True, help="Path to the dataset manifest file.")
    validate.add_argument("--num-classes", type=int, default=None, help="Flag mask labels outside [0, N).")
    validate.add_argument("--ignore-index", type=int, default=None, help="Label value excluded from the check.")
    validate.set_defaults(handler=handlers.dataset_validate)

    inspect = dataset_subparsers.add_parser("inspect", help="Report dataset statistics.")
    inspect.add_argument("--manifest", required=True, help="Path to the dataset manifest file.")
    inspect.set_defaults(handler=handlers.dataset_inspect)

    init = dataset_subparsers.add_parser(
        "init", help="Write an example dataset manifest CSV (placeholder rows) to edit."
    )
    init.add_argument("--output", required=True, help="Destination path for the generated manifest CSV.")
    init.add_argument("--force", action="store_true", help="Overwrite --output if it already exists.")
    init.set_defaults(handler=handlers.dataset_init)


def _add_encoder_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=None, help="Hosted encoder API base URL; omit to use the fake encoder.")
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"Encoder API key; defaults to ${ENCODER_API_KEY_VARIABLE} from the environment or a .env file.",
    )
    parser.add_argument("--model", default="pai-embedding", help="Encoder model alias (default: pai-embedding).")
    parser.add_argument(
        "--preprocessing", default="", help="A stable label for preprocessing params, for cache invalidation."
    )


def _build_config_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("config", help="Generate and inspect training run config files.")
    config_subparsers = parser.add_subparsers(dest="subcommand", required=True)

    init = config_subparsers.add_parser(
        "init", help="Write a fully-commented training config template (placeholders + defaults)."
    )
    init.add_argument("--output", required=True, help="Destination path for the generated YAML file.")
    init.add_argument("--force", action="store_true", help="Overwrite --output if it already exists.")
    init.set_defaults(handler=handlers.config_init)


def _build_encoder_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("encoder", help="Benchmark the remote encoder API.")
    encoder_subparsers = parser.add_subparsers(dest="subcommand", required=True)

    benchmark = encoder_subparsers.add_parser(
        "benchmark", help="Measure throughput/latency and recommend batch size and concurrency."
    )
    _add_encoder_selection_arguments(benchmark)
    benchmark.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16], help="Batch sizes to test.")
    benchmark.add_argument(
        "--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8], help="Concurrency levels to test."
    )
    benchmark.add_argument(
        "--num-requests", type=int, default=10, help="Requests issued per (batch size, concurrency) combination."
    )
    benchmark.add_argument("--image-size", type=int, default=64, help="Side length of the dummy square test image.")
    benchmark.set_defaults(handler=handlers.encoder_benchmark)


def _build_features_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("features", help="Build and manage the offline feature cache.")
    features_subparsers = parser.add_subparsers(dest="subcommand", required=True)

    build = features_subparsers.add_parser("build", help="Resumable offline feature precomputation.")
    build.add_argument("--manifest", required=True, help="Path to the dataset manifest file.")
    build.add_argument("--store", required=True, help="Directory the feature store is (or will be) built in.")
    build.add_argument(
        "--augmentation-config",
        default=None,
        help="Path to a YAML file shaped like configs/augmentation/{none,offline}.yaml (mode/variant/"
        "geometric/photometric); omit for no augmentation. mode must be 'none' or 'offline' — online/"
        "hybrid augmentation cannot be finitely precomputed. Point this at the exact same file the "
        "training config's augmentation: block will use, or the feature cache keys won't match.",
    )
    build.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed offline augmentation derives each sample's seed from — must match the "
        "training config's top-level seed (only consulted when --augmentation-config sets mode: offline).",
    )
    _add_encoder_selection_arguments(build)
    build.set_defaults(handler=handlers.features_build)

    verify = features_subparsers.add_parser("verify", help="Verify feature store integrity.")
    verify.add_argument("--store", required=True, help="Directory the feature store was built in.")
    verify.set_defaults(handler=handlers.features_verify)

    inspect = features_subparsers.add_parser("inspect", help="Report feature store statistics.")
    inspect.add_argument("--store", required=True, help="Directory the feature store was built in.")
    inspect.set_defaults(handler=handlers.features_inspect)

    clean = features_subparsers.add_parser("clean", help="Remove stale or orphaned feature shards.")
    clean.add_argument("--store", required=True, help="Directory the feature store was built in.")
    clean.set_defaults(handler=handlers.features_clean)


def _build_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Train a segmentation decoder.")
    parser.add_argument("--config", required=True, help="Path to a training run YAML config file.")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value / key.nested=value config overrides.")
    parser.set_defaults(handler=handlers.train)


def _add_scored_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, help="Path to the dataset manifest file.")
    parser.add_argument("--store", required=True, help="Directory the feature store was built in.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint (e.g. last.ckpt/best.ckpt).")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of segmentation classes.")
    parser.add_argument(
        "--decoder",
        default="mlp_probe",
        choices=["mlp_probe", "token_fpn", "aspp", "ppm", "segmenter", "mask_former"],
        help="Decoder architecture.",
    )
    parser.add_argument(
        "--decoder-kwargs",
        default="{}",
        help="JSON object of extra decoder constructor kwargs — must match the training run's "
        'decoder_kwargs exactly (see its config.resolved.yaml), e.g. \'{"cls_fusion": "film"}\' '
        "for token_fpn, or the checkpoint's state dict will not load.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--sample-ids", nargs="*", default=None, help="Restrict to these sample IDs; omit for the whole manifest."
    )
    parser.add_argument(
        "--encoder-model",
        default="fake-encoder",
        help="Encoder model alias the feature store was built with (default: the fake encoder's).",
    )
    parser.add_argument(
        "--encoder-revision",
        default="fake-v1",
        help="Encoder revision the feature store was built with (default: the fake encoder's).",
    )
    parser.add_argument("--preprocessing", default="", help="Preprocessing label used when the store was built.")


def _build_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint.")
    _add_scored_run_arguments(parser)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Deterministically resize every image/mask to this size before batching — must match "
        "the checkpoint's training run (its augmentation.geometric.resize). Required whenever the "
        "dataset's images/masks do not already share one native size.",
    )
    parser.add_argument(
        "--loss-name",
        default="ce",
        choices=["ce", "bce", "dice", "ce_dice", "bce_dice"],
        help="Loss to report the reconstructed 'loss' metric under — match the checkpoint's "
        "training config, or the reported value won't be comparable (default: ce).",
    )
    parser.add_argument(
        "--loss-ignore-index", type=int, default=-100, help="Pixel value excluded from the loss (default: -100)."
    )
    parser.add_argument(
        "--loss-ce-weight", type=float, default=1.0, help="Weight of the CE/BCE term in a combined loss."
    )
    parser.add_argument(
        "--loss-dice-weight", type=float, default=1.0, help="Weight of the Dice term in a combined loss."
    )
    parser.add_argument(
        "--augmentation-config",
        default=None,
        help="Path to a YAML file shaped like configs/augmentation/{none,offline}.yaml; omit to evaluate "
        "on each sample's native, unaugmented (optionally --resize'd) image (the default). Set this — "
        "pointed at the exact same file the training run's augmentation: block used — whenever that run "
        "used 'augmentation.mode: offline' with a random_crop: a decoder trained only on small fixed-size "
        "crops has no spatial context beyond a single patch and generalizes poorly to a much larger, "
        "differently-shaped native patch grid it never saw in training, which shows up as "
        "content-independent prediction artifacts, not just lower accuracy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed offline augmentation derives each sample's seed from — must match the "
        "training run's top-level seed (only consulted when --augmentation-config sets mode: offline).",
    )
    parser.set_defaults(handler=handlers.evaluate)


def _build_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("predict", help="Run inference and write predictions/visualizations.")
    _add_scored_run_arguments(parser)
    parser.add_argument("--output", required=True, help="Directory to write prediction PNGs into.")
    parser.add_argument(
        "--overlays", action="store_true", help="Also write a {sample_id}_overlay.png visualization per sample."
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Overlay opacity in [0, 1] (default: 0.5).")
    parser.add_argument(
        "--augmentation-config",
        default=None,
        help="Path to a YAML file shaped like configs/augmentation/{none,offline}.yaml; omit to predict "
        "on each sample's native, unaugmented image (the default). Set this — pointed at the exact same "
        "file the training run's augmentation: block used — whenever that run used 'augmentation.mode: "
        "offline' with a random_crop: a decoder trained only on small fixed-size crops has no spatial "
        "context beyond a single patch and generalizes poorly to a much larger, differently-shaped native "
        "patch grid it never saw in training, which shows up as content-independent prediction artifacts, "
        "not just lower accuracy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed offline augmentation derives each sample's seed from — must match the "
        "training run's top-level seed (only consulted when --augmentation-config sets mode: offline).",
    )
    parser.set_defaults(handler=handlers.predict)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``agritune`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all top-level subcommands (``config``, ``dataset``, ``encoder``, ``features``,
        ``train``, ``evaluate``, ``predict``) registered.
    """
    parser = argparse.ArgumentParser(
        prog="agritune",
        description="Train and evaluate agricultural segmentation decoders on frozen features "
        "from a remote ViT encoder.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for AgriTune loggers (default: INFO).",
    )
    subparsers = parser.add_subparsers(dest="command")
    _build_config_parser(subparsers)
    _build_dataset_parser(subparsers)
    _build_encoder_parser(subparsers)
    _build_features_parser(subparsers)
    _build_train_parser(subparsers)
    _build_evaluate_parser(subparsers)
    _build_predict_parser(subparsers)
    return parser


def _apply_environment_defaults(args: argparse.Namespace) -> None:
    """Fill encoder credentials absent from the command line from the environment."""
    if getattr(args, "api_key", None) is None:
        args.api_key = os.environ.get(ENCODER_API_KEY_VARIABLE)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``agritune`` CLI.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argument vector to parse; defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    load_env_file()

    if args.command is None:
        parser.print_help()
        return 0

    _apply_environment_defaults(args)
    handler: _CommandHandler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
