# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Loads ``agritune train`` configuration via OmegaConf.

A YAML file plus ``key=value`` CLI overrides — the same override syntax Hydra uses (see
``agritune_implementation_plan.md`` §23).

Config-group composition (``defaults:`` lists selecting among named ``dataset``/``encoder``/
``decoder``/... variants) is future work — see ``docs/configuration.md``. For now, one YAML file
plus dotlist overrides is resolved directly into a
:class:`~precisionai.agritune.services.training_service.TrainingRunConfig`.
"""

from typing import Any

from omegaconf import OmegaConf

from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.optimization.optimizers import OptimizerConfig
from precisionai.agritune.optimization.schedulers import SchedulerConfig
from precisionai.agritune.services.training_service import TrainingRunConfig
from precisionai.agritune.tasks.segmentation.losses import SegmentationLossConfig
from precisionai.agritune.training.precision import PrecisionConfig
from precisionai.agritune.training.trainer import TrainerConfig


def load_training_run_config(config_path: str, overrides: list[str] | None = None) -> TrainingRunConfig:
    """Load and resolve a :class:`TrainingRunConfig` from YAML plus CLI overrides.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file (see ``examples/segmentation/`` for the expected shape).
    overrides : list[str] | None, optional
        ``key=value`` / ``key.nested=value`` override strings, e.g. ``["trainer.max_epochs=10"]``.

    Returns
    -------
    TrainingRunConfig
    """
    merged = OmegaConf.load(config_path)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    resolved: dict[str, Any] = OmegaConf.to_container(merged, resolve=True)  # type: ignore[assignment]
    return _config_from_dict(resolved)


def _config_from_dict(data: dict[str, Any]) -> TrainingRunConfig:
    encoder_data = data["encoder_fingerprint"]
    optimizer_data = dict(data.get("optimizer") or {})
    if "betas" in optimizer_data:
        optimizer_data["betas"] = tuple(optimizer_data["betas"])
    scheduler_data = data.get("scheduler")
    trainer_data = dict(data.get("trainer") or {})
    loss_data = dict(data.get("loss") or {})
    seed = data.get("seed", 0)

    return TrainingRunConfig(
        manifest_path=data["manifest_path"],
        feature_store_dir=data["feature_store_dir"],
        run_root=data.get("run_root", "runs"),
        run_id=data["run_id"],
        num_classes=data["num_classes"],
        encoder_fingerprint=EncoderFingerprint(
            model=encoder_data["model"],
            revision=encoder_data.get("revision"),
            preprocessing=encoder_data.get("preprocessing", ""),
        ),
        decoder_name=data.get("decoder_name", "linear"),
        batch_size=data.get("batch_size", 4),
        val_fraction=data.get("val_fraction", 0.2),
        seed=seed,
        optimizer=OptimizerConfig(**optimizer_data),
        scheduler=SchedulerConfig(**scheduler_data) if scheduler_data else None,
        trainer=TrainerConfig(
            max_epochs=trainer_data.get("max_epochs", 1),
            accumulation_steps=trainer_data.get("accumulation_steps", 1),
            grad_clip_norm=trainer_data.get("grad_clip_norm"),
            precision=PrecisionConfig(mode=trainer_data.get("precision", "fp32")),
            seed=trainer_data.get("seed", seed),
            early_stopping_patience=trainer_data.get("early_stopping_patience"),
            checkpoint_every_n_steps=trainer_data.get("checkpoint_every_n_steps"),
        ),
        loss=SegmentationLossConfig(**loss_data),
        val_metric_name=data.get("val_metric_name", "mean_iou"),
        higher_is_better=data.get("higher_is_better", True),
        checkpoint_top_k=data.get("checkpoint_top_k", 3),
    )
