# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precisionai.agritune.tasks.segmentation.postprocessing."""

import torch

from precisionai.agritune.tasks.segmentation.postprocessing import logits_to_predictions, resize_predictions


def test_logits_to_predictions_argmaxes_class_dimension() -> None:
    logits = torch.zeros(1, 3, 2, 2)
    logits[0, 2, 0, 0] = 5.0  # class 2 is the clear winner at (0, 0)
    predictions = logits_to_predictions(logits)
    assert predictions.shape == (1, 2, 2)
    assert predictions[0, 0, 0].item() == 2


def test_resize_predictions_preserves_label_values() -> None:
    predictions = torch.tensor([[[0, 3], [3, 0]]], dtype=torch.long)
    resized = resize_predictions(predictions, (4, 4))
    assert resized.shape == (1, 4, 4)
    assert set(resized.unique().tolist()) <= {0, 3}


def test_resize_predictions_preserves_dtype() -> None:
    predictions = torch.zeros(1, 2, 2, dtype=torch.long)
    resized = resize_predictions(predictions, (2, 2))
    assert resized.dtype == torch.long
