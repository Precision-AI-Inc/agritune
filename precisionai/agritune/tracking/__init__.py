# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Tracker protocol and backends (Null/JSONL/TensorBoard/MLflow/W&B) plus MultiTracker.

External trackers (TensorBoard/MLflow/W&B) are optional extras — nothing in AgriTune requires a
tracking backend, let alone a SaaS one, to train.
"""

from precisionai.agritune.tracking.jsonl import JSONLTracker
from precisionai.agritune.tracking.mlflow_tracker import MLflowTracker
from precisionai.agritune.tracking.multi import MultiTracker
from precisionai.agritune.tracking.null import NullTracker
from precisionai.agritune.tracking.tensorboard import TensorBoardTracker
from precisionai.agritune.tracking.wandb_tracker import WandBTracker

__all__ = [
    "JSONLTracker",
    "MLflowTracker",
    "MultiTracker",
    "NullTracker",
    "TensorBoardTracker",
    "WandBTracker",
]
