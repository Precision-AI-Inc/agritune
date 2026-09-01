# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""``CometTracker`` — an optional Comet ML tracking backend.

Requires the ``comet_ml`` package (``pip install pai-agritune[tracking]``); never required to run
AgriTune otherwise.

Unlike the other optional trackers, ``comet_ml`` is deliberately **not** imported at module level
(a deviation from CLAUDE.md's usual "module-level try/except" pattern for optional deps, and from
the docstring in this same section before it was fixed to match reality). Merely importing
``comet_ml`` activates its automatic instrumentation of other frameworks present in the process
(notably ``mlflow``) — it silently mirrors their logging calls into a hidden offline Comet
experiment, regardless of whether any code ever constructs a Comet ``Experiment``. Since
``precisionai.agritune.services.tracking_selection`` imports every tracker module up front to
build its backend registry, an eager top-level import here would inject an uninvited Comet
experiment into every run that selects a *different* backend (e.g. ``"mlflow"``) whenever
``comet_ml`` merely happens to be installed alongside it — which it always is, since both ship in
the same ``pai-agritune[tracking]`` extra. The import is therefore deferred to the first
:class:`CometTracker` construction, so it only happens when a caller actually asks for Comet.
"""

from typing import Any

comet_ml: Any = None
_COMET_AVAILABLE = False
_comet_import_attempted = False


def _ensure_comet_imported() -> None:
    """Perform the (module-global, one-time) ``comet_ml`` import on first use.

    A no-op after the first call — including when a test has monkeypatched ``_COMET_AVAILABLE``/
    ``comet_ml`` directly, since that already implies an import was "attempted".
    """
    global comet_ml, _COMET_AVAILABLE, _comet_import_attempted  # noqa: PLW0603 — one-time lazy-import cache
    if _comet_import_attempted:
        return
    _comet_import_attempted = True
    try:
        import comet_ml as _comet_ml_module  # noqa: PLC0415 — see module docstring: must not run at import time

        comet_ml = _comet_ml_module
        _COMET_AVAILABLE = True
    except ImportError:
        pass


class CometTracker:
    """Logs metrics/params/artifacts to a Comet ML experiment.

    Parameters
    ----------
    project_name : str
        Comet project name.
    api_key : str | None, optional
        Comet API key; ``None`` uses the ``COMET_API_KEY`` environment variable / Comet config.
    workspace : str | None, optional
        Comet workspace name; ``None`` uses the account's default workspace.

    Raises
    ------
    ImportError
        If the ``comet_ml`` package is not installed.
    """

    def __init__(self, *, project_name: str, api_key: str | None = None, workspace: str | None = None) -> None:
        _ensure_comet_imported()
        if not _COMET_AVAILABLE:
            raise ImportError("comet_ml is required for CometTracker: pip install pai-agritune[tracking]") from None
        self._experiment = comet_ml.Experiment(project_name=project_name, api_key=api_key, workspace=workspace)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Log each metric to the active Comet experiment at ``step``."""
        self._experiment.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log ``params`` to the active Comet experiment."""
        self._experiment.log_parameters(params)

    def log_artifact(self, path: str) -> None:
        """Upload the file at ``path`` as a Comet asset."""
        self._experiment.log_asset(path)

    def close(self) -> None:
        """End the active Comet experiment."""
        self._experiment.end()
