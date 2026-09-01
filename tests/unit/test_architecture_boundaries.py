# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Protect AgriTune's non-negotiable import boundaries with static tests."""

import ast
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parents[2] / "precisionai" / "agritune"
_FORBIDDEN_ENCODER_IMPORT = "precisionai.agritune.encoder"


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def test_training_and_tasks_never_import_encoder_modules() -> None:
    violations: list[str] = []
    for subsystem in ("training", "tasks"):
        for path in (_PACKAGE_ROOT / subsystem).rglob("*.py"):
            forbidden = [module for module in _imported_modules(path) if module.startswith(_FORBIDDEN_ENCODER_IMPORT)]
            if forbidden:
                violations.append(f"{path.relative_to(_PACKAGE_ROOT)}: {forbidden}")

    assert violations == []
