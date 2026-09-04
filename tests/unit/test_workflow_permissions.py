# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Policy regression: pull-request CI must not run candidate code with a write token."""

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_PULL_REQUEST_WORKFLOWS = ("ci.yml", "unit-test.yml", "pre-commit.yml")


def _load_workflow(name: str) -> dict[Any, Any]:
    with (_WORKFLOWS / name).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _workflow_on(workflow: dict[Any, Any]) -> Any:
    """Return the workflow trigger block.

    PyYAML 1.1 may parse the unquoted key ``on`` as boolean ``True``.
    """
    if True in workflow:
        return workflow[True]
    return workflow.get("on")


def _permissions_nodes(node: Any) -> list[Any]:
    found: list[Any] = []
    if isinstance(node, dict):
        if "permissions" in node:
            found.append(node["permissions"])
        for value in node.values():
            found.extend(_permissions_nodes(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_permissions_nodes(value))
    return found


def _assert_permissions_are_read_only(permissions: Any, *, label: str) -> None:
    if permissions is None:
        return
    if isinstance(permissions, str):
        assert permissions in {"read-all", "read"}, f"{label} requests {permissions}"
        return
    assert isinstance(permissions, dict), f"{label} has unexpected permissions: {permissions!r}"
    for scope, access in permissions.items():
        assert access != "write", f"{label} grants {scope}: write"
        assert access != "write-all", f"{label} grants write-all"


def _assert_trigger_includes_pull_request(on_block: Any, *, label: str) -> None:
    assert on_block is not None, f"{label} is missing an on/trigger block"
    if isinstance(on_block, str):
        assert on_block == "pull_request", f"{label} trigger is {on_block!r}"
        return
    if isinstance(on_block, list):
        assert "pull_request" in on_block, f"{label} triggers omit pull_request"
        return
    assert isinstance(on_block, dict), f"{label} has unexpected on block: {on_block!r}"
    assert "pull_request" in on_block, f"{label} triggers omit pull_request"


def _checkout_steps(workflow: dict[Any, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for job in workflow.get("jobs", {}).values():
        if not isinstance(job, dict):
            continue
        steps.extend(
            step
            for step in job.get("steps", [])
            if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/checkout@")
        )
    return steps


def test_pull_request_ci_requests_read_only_contents() -> None:
    workflow = _load_workflow("ci.yml")
    assert workflow.get("permissions") == {"contents": "read"}
    _assert_trigger_includes_pull_request(_workflow_on(workflow), label="ci.yml")


def test_pull_request_jobs_cannot_regain_write_permission() -> None:
    for name in _PULL_REQUEST_WORKFLOWS:
        workflow = _load_workflow(name)
        permissions_nodes = _permissions_nodes(workflow)
        assert permissions_nodes, f"{name} must declare explicit read-only permissions"
        for permissions in permissions_nodes:
            _assert_permissions_are_read_only(permissions, label=name)


def test_pull_request_checkouts_do_not_persist_credentials() -> None:
    for name in ("unit-test.yml", "pre-commit.yml"):
        steps = _checkout_steps(_load_workflow(name))
        assert steps, f"{name} is expected to check out the candidate revision"
        for step in steps:
            persist = (step.get("with") or {}).get("persist-credentials")
            assert persist is False, f"{name} checkout must set persist-credentials: false"
