from dataclasses import dataclass
from typing import Literal

import torch as t


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_ctx: int = 1024
    d_model: int = 768
    d_mlp = 4 * d_model
    n_layer: int = 12
    n_head: int = 12
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    mlp_activation: Literal["GELU", "ReLU", "ELU"] = "GELU"


@dataclass
class TrainerConfig:
    ckpt_path: str
    log_path: str
    device: Literal["cpu", "cuda"] = 'cuda' if t.cuda.is_available() else 'cpu'
    batch_size: int = 64
    lr: float = 6.25e-5
    optimizer: Literal["Adam", "AdamW"] = "Adam"
    n_epochs: int = 1
    warmup_steps: int = 0
    log_interval: int = 100
    ckpt_interval: int = 1000
 
