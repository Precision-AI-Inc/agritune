# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Exception hierarchy shared by every :class:`~precisionai.agritune.schemas.protocols.EncoderBackend`.

The retry policy (``precisionai.agritune.encoder.retry``) branches on these types, not on
backend-specific exceptions (e.g. ``openai.APIStatusError``) — ``RemoteEncoderBackend`` is
responsible for translating SDK/HTTP errors into this hierarchy.
"""


class EncoderError(Exception):
    """Base class for all encoder-related errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    status_code : int | None, optional
        HTTP status code, when the error originated from an HTTP response.
    retry_after_seconds : float | None, optional
        Server-supplied ``Retry-After`` duration, when present. Authoritative over any
        client-side backoff schedule.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class EncoderTimeoutError(EncoderError):
    """The encoder request exceeded its configured timeout."""


class EncoderRateLimitError(EncoderError):
    """The encoder rejected the request with HTTP 429 (quota/rate limit exceeded)."""


class EncoderServerError(EncoderError):
    """The encoder returned a server-side error (HTTP 5xx)."""


class MalformedEncoderResponseError(EncoderError):
    """The encoder returned a response that failed schema/shape validation."""
