# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Exceptions raised by feature providers."""


class FeatureNotCachedError(KeyError):
    """A CachedFeatureProvider found no cached entry for a requested sample.

    Subclasses ``KeyError`` since it is fundamentally a missing-key lookup failure, but is spelled
    out as its own type so callers can give an actionable message (e.g. "run `agritune features
    build` first") without inspecting a bare ``KeyError``'s arguments.
    """
