# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Redact secrets from log records before they are emitted.

Call sites still must not log raw API keys or Authorization headers, but a central filter is the
backstop so a DEBUG override dump or a URL with userinfo cannot print credentials.
"""

from __future__ import annotations

import logging
import re

_REDACTED = "<redacted>"
_SENSITIVE_NAME = r"authorization|api[_-]?key|api[_-]?token|password|secret|token"
_SENSITIVE_ASSIGNMENT = re.compile(rf"(?i)([\w.-]*(?:{_SENSITIVE_NAME}))(\s*[:=]\s*)([^\s,;]+)")
_BEARER_TOKEN = re.compile(r"(?i)(\bbearer\s+)(\S+)")
_URL_USERINFO = re.compile(r"(\w+://)([^/@:\s]+):([^/@\s]+)@")
_QUERY_SECRET = re.compile(rf"(?i)([?&](?:{_SENSITIVE_NAME})=)([^&\s]+)")


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
        return True
