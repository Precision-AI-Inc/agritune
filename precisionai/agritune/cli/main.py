# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The ``agritune`` command-line entry point.

Run ``agritune --help`` to see available commands.

Subcommands are registered here even before their underlying service exists; an unimplemented
subcommand exits with status 1 and a clear message rather than a stack trace. As each phase of
``agritune_implementation_plan.md`` lands, its handler is wired to the corresponding
``precisionai.agritune.services`` call.
"""

import argparse
import sys
from collections.abc import Callable, Sequence

from precisionai.agritune.cli import handlers
from precisionai.agritune.logging import configure_logging, get_logger

logger = get_logger(__name__)

_CommandHandler = Callable[[argparse.Namespace], int]


def _not_implemented(command: str) -> _CommandHandler:
    def handler(_args: argparse.Namespace) -> int:
        print(f"agritune {command}: not implemented yet.", file=sys.stderr)
        return 1

    return handler


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


def _build_encoder_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("encoder", help="Benchmark the remote encoder API.")
    encoder_subparsers = parser.add_subparsers(dest="subcommand", required=True)

    benchmark = encoder_subparsers.add_parser(
        "benchmark", help="Measure throughput/latency and recommend batch size and concurrency."
    )
    benchmark.set_defaults(handler=_not_implemented("encoder benchmark"))


def _add_encoder_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=None, help="Hosted encoder API base URL; omit to use the fake encoder.")
    parser.add_argument("--api-key", default=None, help="Encoder API key (or set via your shell environment).")
    parser.add_argument("--model", default="pai-embedding", help="Encoder model alias (default: pai-embedding).")
    parser.add_argument(
        "--preprocessing", default="", help="A stable label for preprocessing params, for cache invalidation."
    )


def _build_features_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("features", help="Build and manage the offline feature cache.")
    features_subparsers = parser.add_subparsers(dest="subcommand", required=True)

    build = features_subparsers.add_parser("build", help="Resumable offline feature precomputation.")
    build.add_argument("--manifest", required=True, help="Path to the dataset manifest file.")
    build.add_argument("--store", required=True, help="Directory the feature store is (or will be) built in.")
    _add_encoder_selection_arguments(build)
    build.set_defaults(handler=handlers.features_build)

    verify = features_subparsers.add_parser("verify", help="Verify feature store integrity.")
    verify.set_defaults(handler=_not_implemented("features verify"))

    inspect = features_subparsers.add_parser("inspect", help="Report feature store statistics.")
    inspect.set_defaults(handler=_not_implemented("features inspect"))

    clean = features_subparsers.add_parser("clean", help="Remove stale or orphaned feature shards.")
    clean.set_defaults(handler=_not_implemented("features clean"))


def _build_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Train a segmentation decoder.")
    parser.add_argument("--config", required=True, help="Path to a training run YAML config file.")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value / key.nested=value config overrides.")
    parser.set_defaults(handler=handlers.train)


def _build_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint.")
    parser.set_defaults(handler=_not_implemented("evaluate"))


def _build_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("predict", help="Run inference and write predictions/visualizations.")
    parser.set_defaults(handler=_not_implemented("predict"))


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``agritune`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all top-level subcommands (``dataset``, ``encoder``, ``features``, ``train``,
        ``evaluate``, ``predict``) registered.
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
    _build_dataset_parser(subparsers)
    _build_encoder_parser(subparsers)
    _build_features_parser(subparsers)
    _build_train_parser(subparsers)
    _build_evaluate_parser(subparsers)
    _build_predict_parser(subparsers)
    return parser


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

    if args.command is None:
        parser.print_help()
        return 0

    handler: _CommandHandler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
