# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.cli.main."""

import pytest

from precisionai.agritune.cli.main import build_parser, main


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 0
    assert "usage: agritune" in capsys.readouterr().out


def test_help_flag_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_help_lists_all_top_level_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("dataset", "encoder", "features", "train", "evaluate", "predict"):
        assert command in out


@pytest.mark.parametrize(
    "argv",
    [
        ["encoder", "benchmark"],
        ["features", "verify"],
        ["features", "inspect"],
        ["features", "clean"],
        ["evaluate"],
        ["predict"],
    ],
)
def test_unimplemented_commands_exit_nonzero(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(argv)
    assert exit_code == 1
    assert "not implemented yet" in capsys.readouterr().err


def test_dataset_subcommand_requires_manifest() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["dataset", "validate"])
    assert exc_info.value.code == 2


def test_features_build_requires_manifest_and_store() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["features", "build"])
    assert exc_info.value.code == 2


def test_train_requires_config() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["train"])
    assert exc_info.value.code == 2


def test_build_parser_prog_name() -> None:
    parser = build_parser()
    assert parser.prog == "agritune"
