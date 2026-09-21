# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Loads ``agritune train`` configuration via OmegaConf, with two supported shapes.

A plain YAML file (no ``defaults:`` key) is loaded and merged with ``key=value`` CLI overrides
directly — the original, flat shape every example and test in this repository predates this
module's Hydra support with. A YAML file that *does* have a ``defaults:`` key (like the packaged
``precisionai/agritune/configs/config.yaml``) is instead resolved as a genuine Hydra config-group
composition: ``compose()`` selects one variant per group (``dataset``/``encoder``/``augmentation``/
``feature_augmentation``/``feature_provider``/``task``/``decoder``/``optimizer``/``scheduler``/
``tracking``), and the same
``overrides`` list can both override plain fields (``trainer.max_epochs=3``) and swap group
variants (``decoder=token_fpn``, ``augmentation=online``) — see ``docs/configuration.md``.
"""

from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from precisionai.agritune.augmentations.feature.pipeline import FeatureAugmentationConfig
from precisionai.agritune.augmentations.image.pipeline import AugmentationMode, GeometricConfig, PhotometricConfig
from precisionai.agritune.features.keys import EncoderFingerprint
from precisionai.agritune.optimization.optimizers import OptimizerConfig
from precisionai.agritune.optimization.schedulers import SchedulerConfig
from precisionai.agritune.services.tracking_selection import TrackingSelection
from precisionai.agritune.services.training_service import AugmentationSelection, TrainingRunConfig
from precisionai.agritune.tasks.segmentation.losses import SegmentationLossConfig
from precisionai.agritune.training.precision import PrecisionConfig
from precisionai.agritune.training.trainer import TrainerConfig
from precisionai.agritune.utils.env import load_env_file


def load_training_run_config(config_path: str, overrides: list[str] | None = None) -> TrainingRunConfig:
    """Load and resolve a :class:`TrainingRunConfig` from YAML plus CLI overrides.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file. A file with a ``defaults:`` key is resolved via Hydra
        config-group composition (see ``precisionai/agritune/configs/``); any other file is loaded
        and merged with ``overrides`` directly.
    overrides : list[str] | None, optional
        ``key=value`` / ``key.nested=value`` override strings, e.g. ``["trainer.max_epochs=10"]``
        — or, for a Hydra-composed config, a group-variant swap like ``["decoder=token_fpn"]``.

    Returns
    -------
    TrainingRunConfig

    See Also
    --------
    precisionai.agritune.utils.env.load_env_file : Loads a ``.env`` file so that
        ``${oc.env:...}`` interpolations can resolve from it.
    """
    # Must precede resolution: `${oc.env:...}` interpolations read os.environ, so a .env file
    # only reaches them if it has been loaded into the environment by this point.
    load_env_file()

    overrides = overrides or []
    path = Path(config_path)
    raw = OmegaConf.load(path)

    original: dict[str, Any] = OmegaConf.to_container(raw, resolve=False)  # type: ignore[assignment]
    if isinstance(raw, DictConfig) and "defaults" in raw:
        resolved = _compose_config_groups(path, overrides)
    else:
        merged = OmegaConf.merge(raw, OmegaConf.from_dotlist(overrides)) if overrides else raw
        resolved = OmegaConf.to_container(merged, resolve=True)  # type: ignore[assignment]

    config = _config_from_dict(resolved)  # type: ignore[arg-type]
    config.original_config = original
    config.config_overrides = list(overrides)
    return config


def _compose_config_groups(config_path: Path, overrides: list[str]) -> dict[str, Any]:
    config_dir = str(config_path.parent.resolve())
    config_name = config_path.stem
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    # throw_on_missing surfaces a forgotten `???` (e.g. dataset.manifest_path) as a clear
    # MissingMandatoryValue naming the exact key, instead of a literal "???" string silently
    # flowing into TrainingRunConfig and failing confusingly much later.
    composed: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    return _flatten_composed_groups(composed)


def _flatten_composed_groups(composed: dict[str, Any]) -> dict[str, Any]:
    """Fold group-namespaced Hydra output into the flat shape :func:`_config_from_dict` expects.

    ``optimizer``/``scheduler``/``augmentation``/``feature_augmentation``/``tracking`` are already
    nested dicts in that flat shape, so those five groups pass through unchanged; ``dataset``/
    ``encoder``/``decoder``/``task``/``feature_provider`` do not have a flat-shape equivalent and
    are unpacked here.
    """
    flat = dict(composed)
    dataset = flat.pop("dataset", {}) or {}
    encoder = flat.pop("encoder", {}) or {}
    decoder = flat.pop("decoder", {}) or {}
    task = flat.pop("task", {}) or {}
    feature_provider = flat.pop("feature_provider", {}) or {}

    flat["manifest_path"] = dataset.get("manifest_path")
    flat["num_classes"] = dataset.get("num_classes")
    flat["encoder_fingerprint"] = {
        "model": encoder.get("model"),
        "revision": encoder.get("revision"),
        "preprocessing": encoder.get("preprocessing", ""),
    }
    flat["encoder_base_url"] = encoder.get("base_url")
    flat["encoder_api_key"] = encoder.get("api_key")
    flat["decoder_name"] = decoder.get("name", "mlp_probe")
    flat["decoder_kwargs"] = {key: value for key, value in decoder.items() if key != "name"}
    flat["loss"] = task.get("loss", {})
    flat["val_metric_name"] = task.get("val_metric_name", "mean_iou")
    flat["higher_is_better"] = task.get("higher_is_better", True)
    flat["feature_provider"] = feature_provider.get("name", "cached")
    return flat


def _tupleize(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    result = dict(data)
    for key in keys:
        if result.get(key) is not None:
            result[key] = tuple(result[key])
    return result


def load_augmentation_selection(config_path: str | None) -> AugmentationSelection:
    """Load an :class:`AugmentationSelection` from a YAML file shaped like ``configs/augmentation/*.yaml``.

    Shared by ``agritune features build --augmentation-config`` and ``POST /features/build``'s
    ``augmentation_config_path``, so an offline-augmented feature cache and the training run that
    reads it can point at the exact same file rather than duplicating (and risking a mismatched
    copy of) the ``geometric``/``photometric`` parameters — see ``docs/feature-caching.md``.

    Parameters
    ----------
    config_path : str | None
        Path to a YAML file with top-level ``mode``/``variant``/``geometric``/``photometric`` keys
        (e.g. ``precisionai/agritune/configs/augmentation/offline.yaml``); ``None`` returns the
        default selection (``mode: none`` — no augmentation).

    Returns
    -------
    AugmentationSelection
    """
    if config_path is None:
        return AugmentationSelection()
    data: dict[str, Any] = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)  # type: ignore[assignment]
    return _augmentation_from_dict(data)


def _augmentation_from_dict(data: dict[str, Any]) -> AugmentationSelection:
    geometric_data = _tupleize(dict(data.get("geometric") or {}), "resize", "random_crop")
    photometric_data = _tupleize(
        dict(data.get("photometric") or {}),
        "brightness_range",
        "contrast_range",
        "saturation_range",
        "blur_radius_range",
        "noise_std_range",
        "gsd_jitter_scale_range",
        "vegetation_index_jitter_range",
    )
    return AugmentationSelection(
        mode=AugmentationMode(data.get("mode", "none")),
        variant=data.get("variant", 0),
        hybrid_online_probability=data.get("hybrid_online_probability", 0.3),
        geometric=GeometricConfig(**geometric_data),
        photometric=PhotometricConfig(**photometric_data),
    )


def _feature_augmentation_from_dict(data: dict[str, Any]) -> FeatureAugmentationConfig:
    return FeatureAugmentationConfig(
        patch_dropout_probability=data.get("patch_dropout_probability", 0.0),
        token_masking_probability=data.get("token_masking_probability", 0.0),
        token_mask_value=data.get("token_mask_value", 0.0),
        gaussian_noise_std=data.get("gaussian_noise_std", 0.0),
        cls_dropout_probability=data.get("cls_dropout_probability", 0.0),
        channel_dropout_probability=data.get("channel_dropout_probability", 0.0),
    )


def _tracking_from_dict(data: dict[str, Any]) -> TrackingSelection:
    return TrackingSelection(
        backends=list(data.get("backends", ["jsonl"])),
        mlflow_experiment_name=data.get("mlflow_experiment_name"),
        mlflow_tracking_uri=data.get("mlflow_tracking_uri"),
        wandb_project=data.get("wandb_project", "agritune"),
        neptune_project=data.get("neptune_project"),
        comet_project_name=data.get("comet_project_name", "agritune"),
    )


def _config_from_dict(data: dict[str, Any]) -> TrainingRunConfig:
    encoder_data = data["encoder_fingerprint"]
    optimizer_data = dict(data.get("optimizer") or {})
    if "betas" in optimizer_data:
        optimizer_data["betas"] = tuple(optimizer_data["betas"])

    scheduler_data = dict(data.get("scheduler") or {})
    scheduler_enabled = scheduler_data.pop("enabled", True)

    trainer_data = dict(data.get("trainer") or {})
    loss_data = dict(data.get("loss") or {})
    if loss_data.get("class_weights") is not None:
        loss_data["class_weights"] = torch.tensor(loss_data["class_weights"], dtype=torch.float32)
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
        feature_provider=data.get("feature_provider", "cached"),
        store_type=data.get("store_type", "directory"),
        entries_per_shard=data.get("entries_per_shard", 1000),
        encoder_base_url=data.get("encoder_base_url"),
        encoder_api_key=data.get("encoder_api_key"),
        augmentation=_augmentation_from_dict(dict(data.get("augmentation") or {})),
        feature_augmentation=_feature_augmentation_from_dict(dict(data.get("feature_augmentation") or {})),
        decoder_name=data.get("decoder_name", "mlp_probe"),
        decoder_kwargs=dict(data.get("decoder_kwargs") or {}),
        batch_size=data.get("batch_size", 4),
        device=data.get("device", "cpu"),
        num_workers=data.get("num_workers", 0),
        pin_memory=data.get("pin_memory", False),
        prefetch_factor=data.get("prefetch_factor"),
        feature_read_workers=data.get("feature_read_workers", 32),
        val_fraction=data.get("val_fraction", 0.2),
        seed=seed,
        optimizer=OptimizerConfig(**optimizer_data),
        scheduler=SchedulerConfig(**scheduler_data) if scheduler_data and scheduler_enabled else None,
        trainer=TrainerConfig(
            max_epochs=trainer_data.get("max_epochs", 1),
            accumulation_steps=trainer_data.get("accumulation_steps", 1),
            grad_clip_norm=trainer_data.get("grad_clip_norm"),
            precision=PrecisionConfig(mode=trainer_data.get("precision", "fp32")),
            seed=trainer_data.get("seed", seed),
            early_stopping_patience=trainer_data.get("early_stopping_patience"),
            checkpoint_every_n_steps=trainer_data.get("checkpoint_every_n_steps"),
            strict_resume=trainer_data.get("strict_resume", True),
        ),
        loss=SegmentationLossConfig(**loss_data),
        val_metric_name=data.get("val_metric_name", "mean_iou"),
        higher_is_better=data.get("higher_is_better", True),
        checkpoint_top_k=data.get("checkpoint_top_k", 3),
        tracking=_tracking_from_dict(dict(data.get("tracking") or {})),
    )
