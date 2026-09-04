# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Policy regression: pull-request CI must not run candidate code with a write token."""

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_PULL_REQUEST_WORKFLOWS = ("ci.yml", "unit-test.yml", "pre-commit.yml")


def _load_workflow(name: str) -> dict[str, Any]:
    with (_WORKFLOWS / name).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


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


def _checkout_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
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
    # GitHub's `on:` key is boolean True under PyYAML 1.1; assert the event on the source text.
    text = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "pull_request:" in text


def test_pull_request_jobs_cannot_regain_write_permission() -> None:
    for name in _PULL_REQUEST_WORKFLOWS:
        workflow = _load_workflow(name)
        for permissions in _permissions_nodes(workflow):
            _assert_permissions_are_read_only(permissions, label=name)
        text = (_WORKFLOWS / name).read_text(encoding="utf-8")
        assert "contents: write" not in text
        assert "write-all" not in text


def test_pull_request_checkouts_do_not_persist_credentials() -> None:
    for name in ("unit-test.yml", "pre-commit.yml"):
        steps = _checkout_steps(_load_workflow(name))
        assert steps, f"{name} is expected to check out the candidate revision"
        for step in steps:
            persist = (step.get("with") or {}).get("persist-credentials")
            assert persist is False, f"{name} checkout must set persist-credentials: false"
