# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from precisionai.agritune.cli.main import _apply_environment_defaults
from precisionai.agritune.utils import env


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.ENV_FILE_VARIABLE, raising=False)
    monkeypatch.delenv(env.ENCODER_API_KEY_VARIABLE, raising=False)


def _write_env_file(directory: Path, value: str = "sk-from-file") -> Path:
    path = directory / ".env"
    path.write_text(f"{env.ENCODER_API_KEY_VARIABLE}={value}\n", encoding="utf-8")
    return path


def test_discovers_env_file_in_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _write_env_file(tmp_path)
    monkeypatch.chdir(tmp_path)

    loaded = env.load_env_file()

    assert loaded == expected


def test_discovered_env_file_populates_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env_file(tmp_path)
    monkeypatch.chdir(tmp_path)

    env.load_env_file()

    assert os.environ[env.ENCODER_API_KEY_VARIABLE] == "sk-from-file"


def test_discovery_walks_up_to_a_parent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _write_env_file(tmp_path)
    nested = tmp_path / "runs" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert env._discover_env_file() == expected


def test_exported_variable_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env.ENCODER_API_KEY_VARIABLE, "sk-from-shell")

    env.load_env_file()

    assert os.environ[env.ENCODER_API_KEY_VARIABLE] == "sk-from-shell"


def test_explicit_path_is_loaded(tmp_path: Path) -> None:
    path = _write_env_file(tmp_path, "sk-explicit")

    loaded = env.load_env_file(path)

    assert loaded == path
    assert os.environ[env.ENCODER_API_KEY_VARIABLE] == "sk-explicit"


def test_env_file_variable_selects_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_env_file(tmp_path, "sk-via-variable")
    monkeypatch.setenv(env.ENV_FILE_VARIABLE, str(path))

    assert env.load_env_file() == path
    assert os.environ[env.ENCODER_API_KEY_VARIABLE] == "sk-via-variable"


def test_missing_explicit_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="env file not found"):
        env.load_env_file(tmp_path / "absent.env")


def test_discover_returns_none_when_ancestry_has_no_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty-dir"
    empty.mkdir()

    class _IsolatedCwd:
        def resolve(self) -> "_IsolatedCwd":
            return self

        @property
        def parents(self) -> tuple[Path, ...]:
            return ()

        def __truediv__(self, name: str) -> Path:
            return empty / name

    monkeypatch.setattr(env.Path, "cwd", lambda *args, **kwargs: _IsolatedCwd())
    assert env._discover_env_file() is None


def test_returns_none_when_no_env_file_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "_discover_env_file", lambda: None)

    assert env.load_env_file() is None


def test_discovered_file_is_ignored_without_python_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env, "_DOTENV_AVAILABLE", False)

    assert env.load_env_file() is None
    assert env.ENCODER_API_KEY_VARIABLE not in os.environ


def test_explicit_path_without_python_dotenv_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_env_file(tmp_path)
    monkeypatch.setattr(env, "_DOTENV_AVAILABLE", False)

    with pytest.raises(ImportError, match="python-dotenv is required"):
        env.load_env_file(path)


def test_omegaconf_interpolation_resolves_from_the_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: `${oc.env:...}` has no .env support of its own, only os.environ."""
    _write_env_file(tmp_path, "sk-interpolated")
    monkeypatch.chdir(tmp_path)
    config = OmegaConf.create({"api_key": f"${{oc.env:{env.ENCODER_API_KEY_VARIABLE},null}}"})

    env.load_env_file()

    assert config.api_key == "sk-interpolated"


def test_cli_fills_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.ENCODER_API_KEY_VARIABLE, "sk-from-env")
    args = argparse.Namespace(api_key=None)

    _apply_environment_defaults(args)

    assert args.api_key == "sk-from-env"


def test_cli_api_key_argument_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.ENCODER_API_KEY_VARIABLE, "sk-from-env")
    args = argparse.Namespace(api_key="sk-from-flag")

    _apply_environment_defaults(args)

    assert args.api_key == "sk-from-flag"
