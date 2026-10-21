"""Typed configuration sections. Default experiments are described in default.yaml."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    dataset: str = 'weather'
    stations: list[str] = field(default_factory=lambda: ['weather'])
    num_workers: int = 0
    dataset_root: str | None = None


@dataclass
class ModelConfig:
    net: str = 'ts'
    pretrain: bool = False
    enc_layers: int = 1
    hidden_dim: int = 128
    nheads: int = 8
    dim_feedforward: int = 256
    dropout: float = 0.1
    pre_norm: bool = False
    position_embedding: str = 'learned'
    use_tf: bool = True
    quant_mode: str = 'tanh'
    quant_scale_min: float = 1.0
    quant_scale_max: float = 8.0
    use_deriv: bool = True
    use_ch_attn: bool = True


@dataclass
class TrainingConfig:
    arch: str = 'query_opt'
    batch_size: int = 256
    optim: str = 'AdamW'
    lr: float = 0.001
    wd: float = 0.01
    momen: float = 0.9
    nesterov: bool = True
    gpu: int | None = 0
    code_length: list[int] = field(default_factory=lambda: [96])
    num_samples: int = 2000
    max_epoch: int = 5
    max_iter: int = 120
    early_stop_patience: int = 10
    early_stop_min_iter: int = 40
    lr_step: list[int] = field(default_factory=lambda: [90])
    lr_gamma: float = 0.1
    val_freq: int = 5
    topk: int = -1
    seed: int = 42
    device: str | None = None
    cuda_visible_devices: str = '0,1'


@dataclass
class LossConfig:
    gamma: int = 200
    use_center: bool = True
    center_weight: float = 1.0
    center_margin_mode: str = 'severity_freq'
    center_m0: float = 0.15
    center_lambda: float = 0.5
    center_gamma: float = 0.3
    center_s: float = 12.0
    center_lr: float = 0.001
    use_tfc: bool = True
    tfc_weight: float = 0.5
    use_ord: bool = True
    ord_weight: float = 0.1
    ord_margin: float = 0.1
    ord_smargin: float = 0.05
    ord_neg_boost: float = 2.0
    ord_warmup: int = 30
    use_bitreg: bool = False
    bit_balance_weight: float = 0.1
    bit_decorr_weight: float = 0.1


@dataclass
class EvaluationConfig:
    use_val_select: bool = True
    ckpt: str | None = None
    datadir: str | None = None
    batch_size: int = 512
    k: list[int] = field(default_factory=lambda: [10, 50, 100])
    outdir: str | None = None
    device: str | None = None
    code_length: int = 96


@dataclass
class OutputConfig:
    info: str = 'weather_ts'
    ckpt_root: str = 'checkpoints'
    ckpt_name: str | None = None
    log_root: str = 'logs'


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
