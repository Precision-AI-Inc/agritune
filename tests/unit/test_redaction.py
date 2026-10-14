# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.logging.redaction."""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from precisionai.agritune.cli import handlers
from precisionai.agritune.logging import RedactingFilter, configure_logging, get_logger, redact_text
from precisionai.agritune.services.encoder_selection import build_raw_encoder

_SENTINEL_API_KEY = "sk-sentinel-api-key-DO-NOT-PRINT"
_SENTINEL_AUTH = "sentinel-authorization-token-DO-NOT-PRINT"
_SENTINEL_PASSWORD = "sentinel-url-password-DO-NOT-PRINT"


def test_redact_text_replaces_api_key_assignments() -> None:
    text = redact_text(f"encoder_api_key={_SENTINEL_API_KEY}")
    assert _SENTINEL_API_KEY not in text
    assert "<redacted>" in text


def test_redact_text_replaces_spaced_api_key_phrases() -> None:
    text = redact_text(f"Incorrect API key provided: {_SENTINEL_API_KEY}")
    assert _SENTINEL_API_KEY not in text
    assert "<redacted>" in text


def test_redact_text_replaces_authorization_bearer_values() -> None:
    text = redact_text(f"Authorization: Bearer {_SENTINEL_AUTH}")
    assert _SENTINEL_AUTH not in text
    assert "<redacted>" in text


def test_redact_text_replaces_url_userinfo() -> None:
    text = redact_text(f"https://user:{_SENTINEL_PASSWORD}@encoder.example/v1")
    assert _SENTINEL_PASSWORD not in text
    assert "<redacted>" in text


def test_redact_text_replaces_url_query_credentials() -> None:
    text = redact_text(f"https://encoder.example/v1?api_key={_SENTINEL_API_KEY}")
    assert _SENTINEL_API_KEY not in text
    assert "<redacted>" in text


def test_redact_text_replaces_sensitive_override_dumps() -> None:
    text = redact_text(repr([f"encoder.api_key={_SENTINEL_API_KEY}"]))
    assert _SENTINEL_API_KEY not in text
    assert "<redacted>" in text


def test_redacting_filter_falls_back_when_message_interpolation_fails() -> None:
    record = logging.LogRecord(
        name="agritune.test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="encoder_api_key=%s %s",
        args=(_SENTINEL_API_KEY,),
        exc_info=None,
    )
    assert RedactingFilter().filter(record) is True
    assert _SENTINEL_API_KEY not in record.getMessage()


def test_redacting_filter_redacts_exception_text() -> None:
    record = logging.LogRecord(
        name="agritune.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="request failed",
        args=(),
        exc_info=None,
    )
    record.exc_text = f"Authorization: Bearer {_SENTINEL_AUTH}"
    assert RedactingFilter().filter(record) is True
    assert record.exc_text is not None
    assert _SENTINEL_AUTH not in record.exc_text
    assert "<redacted>" in record.exc_text


def test_redacting_filter_strips_interpolated_sentinel_from_log_record() -> None:
    record = logging.LogRecord(
        name="agritune.test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="dispatching train: overrides=%s",
        args=([f"encoder_api_key={_SENTINEL_API_KEY}"],),
        exc_info=None,
    )
    assert RedactingFilter().filter(record) is True
    message = record.getMessage()
    assert _SENTINEL_API_KEY not in message
    assert "<redacted>" in message


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_configure_logging_redacts_sentinels_at_every_level(level: str, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level)
    logger = get_logger("redaction")
    log = getattr(logger, level.lower())
    log(
        "encoder_api_key=%s Authorization: Bearer %s url=%s",
        _SENTINEL_API_KEY,
        _SENTINEL_AUTH,
        f"https://user:{_SENTINEL_PASSWORD}@encoder.example/v1?api_key={_SENTINEL_API_KEY}",
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert _SENTINEL_API_KEY not in combined
    assert _SENTINEL_AUTH not in combined
    assert _SENTINEL_PASSWORD not in combined


def test_configure_logging_redacts_gateway_style_api_key_message(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING")
    logger = get_logger("encoder.gateway")
    error = RuntimeError(f"Incorrect API key provided: {_SENTINEL_API_KEY}")
    logger.warning(
        "encoder request failed on attempt %d (%s: %s); retrying with backoff",
        1,
        type(error).__name__,
        error,
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert _SENTINEL_API_KEY not in combined
    assert "API key" in captured.err
    assert "<redacted>" in captured.err


def test_configure_logging_redacts_sentinel_from_logger_exception(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("ERROR")
    logger = get_logger("features.precompute")
    try:
        raise RuntimeError(
            f"Authorization: Bearer {_SENTINEL_AUTH}; "
            f"https://user:{_SENTINEL_PASSWORD}@encoder.example/v1?api_key={_SENTINEL_API_KEY}"
        )
    except RuntimeError:
        logger.exception("failed to encode sample %s", "sample-1")
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert _SENTINEL_API_KEY not in combined
    assert _SENTINEL_AUTH not in combined
    assert _SENTINEL_PASSWORD not in combined
    assert "failed to encode sample" in captured.err


def test_configure_logging_installs_redacting_filter_on_the_console_handler() -> None:
    configure_logging("INFO")
    installed = logging.getLogger("agritune").handlers
    assert len(installed) == 1
    assert any(isinstance(item, RedactingFilter) for item in installed[0].filters)


def test_cli_train_debug_log_redacts_override_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("DEBUG")
    monkeypatch.setattr(
        handlers,
        "load_training_run_config",
        lambda _config, _overrides: SimpleNamespace(
            feature_store_dir=str(tmp_path), store_type="directory", entries_per_shard=1000
        ),
    )
    monkeypatch.setattr(handlers, "build_feature_store", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        handlers,
        "run_training",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_directory=SimpleNamespace(path=tmp_path / "run"),
            final_train_state={"epoch": 0},
            train_metrics={},
            val_metrics={},
        ),
    )
    handlers.train(argparse.Namespace(config="train.yaml", overrides=[f"encoder_api_key={_SENTINEL_API_KEY}"]))
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert _SENTINEL_API_KEY not in combined
    assert "dispatching train" in captured.err


def test_remote_encoder_selection_does_not_print_url_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    build_raw_encoder(
        base_url=f"https://user:{_SENTINEL_PASSWORD}@encoder.example/v1?api_key={_SENTINEL_API_KEY}",
        api_key=_SENTINEL_API_KEY,
        model="pai-embedding",
        preprocessing="",
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert _SENTINEL_API_KEY not in combined
    assert _SENTINEL_PASSWORD not in combined
    assert "RemoteEncoderBackend" in captured.err
