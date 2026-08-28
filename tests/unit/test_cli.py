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
        ["dataset", "validate"],
        ["dataset", "inspect"],
        ["features", "build"],
        ["features", "verify"],
        ["features", "inspect"],
        ["features", "clean"],
        ["train"],
        ["evaluate"],
        ["predict"],
    ],
)
def test_commands_require_their_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2


def test_build_parser_prog_name() -> None:
    parser = build_parser()
    assert parser.prog == "agritune"
