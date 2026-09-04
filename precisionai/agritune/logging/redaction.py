# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Redact secrets from log records before they are emitted.

Call sites still must not log raw API keys or Authorization headers, but a central filter is the
backstop so a DEBUG override dump or a URL with userinfo cannot print credentials.
"""

from __future__ import annotations

import logging
import re
from typing import Any

_REDACTED = "<redacted>"
# Allow whitespace around optional separators so OpenAI-style "Incorrect API key provided: sk-…"
# and dotted override names like encoder.api_key=… are both treated as sensitive labels.
_SENSITIVE_LABEL = r"authorization|api\s*[_-]?\s*key|api\s*[_-]?\s*token|password|secret|token"
_SENSITIVE_QUERY_NAME = r"authorization|api[_-]?key|api[_-]?token|password|secret|token"
# Optional short phrase between the label and the separator covers "API key provided: <value>".
_SENSITIVE_ASSIGNMENT = re.compile(rf"(?i)([\w.-]*{_SENSITIVE_LABEL})((?:\s+\w+){{0,3}}\s*[:=]\s*)([^\s,;'\"\]]+)")
_BEARER_TOKEN = re.compile(r"(?i)(\bbearer\s+)(\S+)")
_URL_USERINFO = re.compile(r"(\w+://)([^/@:\s]+):([^/@\s]+)@")
_QUERY_SECRET = re.compile(rf"(?i)([?&](?:{_SENSITIVE_QUERY_NAME})=)([^&\s]+)")


def redact_text(text: str) -> str:
    """Return ``text`` with API keys, Authorization values, and URL credentials replaced.

    Parameters
    ----------
    text : str
        A log message, override dump, or URL that may contain secrets.

    Returns
    -------
    str
        The same text with sensitive values replaced by ``<redacted>``.
    """
    redacted = _BEARER_TOKEN.sub(rf"\1{_REDACTED}", text)
    redacted = _SENSITIVE_ASSIGNMENT.sub(rf"\1\2{_REDACTED}", redacted)
    redacted = _URL_USERINFO.sub(rf"\1{_REDACTED}:{_REDACTED}@", redacted)
    return _QUERY_SECRET.sub(rf"\1{_REDACTED}", redacted)


def _redact_exception_chain(exc: BaseException | None, *, _seen: set[int] | None = None) -> None:
    """Replace sensitive substrings in ``exc`` args and its cause/context chain in place."""
    if exc is None:
        return
    seen = _seen if _seen is not None else set()
    exc_id = id(exc)
    if exc_id in seen:
        return
    seen.add(exc_id)

    if exc.args:
        redacted_args: list[Any] = []
        for arg in exc.args:
            if isinstance(arg, str):
                redacted_args.append(redact_text(arg))
            elif isinstance(arg, BaseException):
                _redact_exception_chain(arg, _seen=seen)
                redacted_args.append(arg)
            else:
                redacted_args.append(arg)
        exc.args = tuple(redacted_args)

    _redact_exception_chain(exc.__cause__, _seen=seen)
    if exc.__context__ is not exc.__cause__:
        _redact_exception_chain(exc.__context__, _seen=seen)


class RedactingFilter(logging.Filter):
    """Replace sensitive substrings on a log record before a handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact ``record`` in place and allow it to be emitted.

        Parameters
        ----------
        record : logging.LogRecord
            The record about to be handled.

        Returns
        -------
        bool
            Always ``True``; redaction never drops the record.
        """
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = str(record.msg)
        record.msg = redact_text(message)
        record.args = ()
        if isinstance(record.exc_text, str):
            record.exc_text = redact_text(record.exc_text)
        # logging.Formatter and RichHandler both render exc_info after filters run, so scrub the
        # live exception objects here before either path formats a traceback.
        if record.exc_info and record.exc_info[1] is not None:
            _redact_exception_chain(record.exc_info[1])
        return True
