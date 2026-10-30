"""Pydantic schema for PtychoFM YAML files.

Partial configurations and extension keys are supported. Use
``model_dump(exclude_unset=True)`` at dictionary/serialization boundaries: missing
values must stay missing for context-dependent runtime defaults (for example,
pretrained encoder sizes, validation batch size, and dataset normalization).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra='allow', validate_assignment=True)


class DataConfig(ConfigModel):
    data_path: str | list[str] | None = None
    packed: bool = False
    test_path: str | None = None
    test_dataset_object_name: str | None = None
    scale: float | None = None
    apply_noise: bool = True
    deterministic_noise: bool = False
    noise_seed: int = 0
    test_apply_noise: bool = False
    normalization_dict_path: str | None = None
    test_normalization: str | None = None
    default_normalization: float | None = None
    cache_object: bool | None = None
    cache_memory_budget_mb: int = 512
    max_probe_modes: int = 8
    max_OPR_modes: int = 1
    train_split: float | None = None
    random_seed: int | None = None
    subset_fraction: float | None = None
    sharding_strategy: str = 'static'
    max_files: int | None = None
    max_shards: int | None = None
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False
    use_cuda_prefetcher: bool = True


class SchedulerConfig(ConfigModel):
    enabled: bool = False
    scheduler_class: str | None = None
    kwargs: dict[str, Any] | None = Field(default_factory=dict)


class SubsettingScheduleConfig(ConfigModel):
    enabled: bool = False
    epochs: list[int] = Field(default_factory=list)
    fractions: list[float] = Field(default_factory=list)


class LossConfig(ConfigModel):
    loss_type: str = 'mse'
    threshold: float = 0.0
    alpha: float = 1.0


class ProbeAwareLossConfig(LossConfig):
    envelope_floor: float = 0.05
    eps: float = 1e-6


class QDependentLossConfig(LossConfig):
    q_beta: float = 1.0
    q_floor: float = 0.05
    log_error: bool = True
    normalize_q: bool = True
    eps: float = 1e-6


class CombinedLossConfig(ConfigModel):
    type: str
    weight: float = 1.0


class TrainingConfig(ConfigModel):
    mode: str | None = None
    batch_size: int | None = None
    valid_batch_size: int | None = None
    learning_rate: float | None = None
    encoder_lr: float | None = None
    amp_decoder_lr: float | None = None
    ph_decoder_lr: float | None = None
    coupled_decoder_lr: float | None = None
    epochs: int | None = None
    loss_function: str | None = None
    weighted_loss: LossConfig = Field(default_factory=LossConfig)
    probe_aware_loss: ProbeAwareLossConfig = Field(default_factory=ProbeAwareLossConfig)
    q_dependent_loss: QDependentLossConfig = Field(default_factory=QDependentLossConfig)
    combined_loss: list[CombinedLossConfig] = Field(default_factory=list)
    lr_scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    data_subsetting_schedule: SubsettingScheduleConfig | None = None
    skip_batch_if_grad_norm_greater_than: float | None = None
    log_every: int = 0
    log_at_flops: list[float] = Field(default_factory=list)
    max_iters: int | None = None
    track_flops: bool = False
    stop_at_last_flop_target: bool = False
    validation_plot_freq: int | None = None
    validation_plot_num_objects: int | None = 1
    validation_plot_central_crop: int | None = None
    validation_plot_object_crop: int | None = None
    test_plot_freq: int | None = None
    test_plot_central_crop: int = 64
    test_plot_object_crop: int = 180
    resume_from_checkpoint: bool = False
    save_epoch_models: bool = False
    checkpoint_freq: int = 0
    finetune_from_model: str | None = None
    platform: str = 'slurm'
    profile: bool = False


class EncoderConfig(ConfigModel):
    img_size: int = 256
    # These defaults depend on encoder_type and are resolved by the model.
    patch_size: int | None = None
    embed_dim: int | None = None
    depth: int | None = None
    num_heads: int | None = None
    in_channels: int = 1
    mlp_ratio: float = 4.0
    use_cls_token: bool = False
    dropout: float = 0.1
    attn_dropout: float = 0.0
    timm_model_name: str = 'vit_large_patch32_224'
    timm_kwargs: dict[str, Any] | None = None
    checkpoint_path: str | None = None
    checkpoint_kwargs: dict[str, Any] = Field(default_factory=dict)
    strict_load: bool | None = None


class DecoderConfig(ConfigModel):
    base_channels: int = 64
    latent_dim: int | None = None
    num_stages: int = 4
    use_batchnorm: bool = True
    dropout: float = 0.1
    blocks_per_stage: int = 2
    phase_activation: str | None = 'custom'
    circular_phase: bool = False


class KaimingConfig(ConfigModel):
    distribution: str = 'uniform'
    mode: str = 'fan_in'
    nonlinearity: str = 'relu'
    a: float = 0.0


class InitConfig(ConfigModel):
    enabled: bool = False
    method: str = 'trunc_normal'
    mean: float = 0.0
    std: float = 0.02
    apply_to: str | list[str] = Field(default_factory=lambda: ['encoder', 'decoders'])
    override_pretrained: bool = False
    kaiming: KaimingConfig | None = None


class OutputNormConfig(ConfigModel):
    # Scale defaults differ between separate and coupled decoders.
    amp_scale: float | None = None
    amp_offset: float = 1.0
    ph_scale: float | None = None
    real_scale: float = 0.1
    real_offset: float = 1.0
    imag_scale: float = 0.1
    imag_offset: float = 0.0


class ModelConfig(ConfigModel):
    encoder_type: str = 'custom'
    decoder_type: str = 'standard'
    coupled_decoder_mode: str | None = None
    subtract_probe_intensity: bool = False
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    output_norm: OutputNormConfig | None = None
    scaling: dict[str, float] = Field(default_factory=dict)


class PathsConfig(ConfigModel):
    model_save_path: str | None = None


class TrainerConfig(ConfigModel):
    run_name: str | int | None = None
    run_num: str | int | None = None


class TrackingConfig(ConfigModel):
    dataset_name: str | None = None
    notes: str | None = None
    log_parameters: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_system_metrics: bool = True
    log_every_n_batches: int | None = 50


class WandbConfig(ConfigModel):
    enabled: bool = False
    entity: str | None = None
    project: str = 'ptycho-fm'
    dataset_name: str | None = None
    notes: str | None = None


class MLflowConfig(ConfigModel):
    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str | None = 'ptycho-fm'
    tags: dict[str, Any] | None = Field(default_factory=dict)
    azureml_compat: bool = False


class PtychoFMConfig(ConfigModel):
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    tracking: TrackingConfig | None = None
    wandb: WandbConfig | None = None
    mlflow: MLflowConfig | None = None
