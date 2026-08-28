# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""The full set of supported optimizer/scheduler names — for CLI help text and config validation."""

from typing import get_args

from precisionai.agritune.optimization.optimizers import OptimizerName
from precisionai.agritune.optimization.schedulers import SchedulerName

OPTIMIZER_NAMES: tuple[str, ...] = get_args(OptimizerName)
SCHEDULER_NAMES: tuple[str, ...] = get_args(SchedulerName)
