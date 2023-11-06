from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel
import torch as t


class GPT2Config(BaseModel):
    vocab_size: int = 50257
    n_ctx: int = 1024
    d_model: int = 768
    d_mlp: int = 4 * d_model
    n_layer: int = 12
    n_head: int = 12
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    mlp_activation: Literal["GELU", "ReLU", "ELU"] = "GELU"


class TrainerConfig(BaseModel):
    dataset: Literal["Skylion007/openwebtext", "stas/openwebtext-10k", "debug"]
    valid_dataset: Optional[str] = None
    ckpt_path: str = './logs'
    log_path: str = './logs'
    device: Literal["cpu", "cuda"] = 'cuda' if t.cuda.is_available() else 'cpu'
    n_workers: int = 4
    batch_size: int = 64
    lr: float = 6.25e-5
    optimizer: Literal["Adam", "AdamW"] = "Adam"
    n_epochs: int = 1
    warmup_steps: int = 0
    log_interval: int = 100
    ckpt_interval: int = 1000
    valid_interval: int = 1000
    # training optimization
    gradient_checkpoint: bool = False
    bfloat16: bool = False
    gradient_accumulation_steps: Optional[int] = None
    # compiled model does not work with gradient checkpointing as of now:
    # https://github.com/pytorch/pytorch/issues/97077
    #compile_model: bool = False
    attention_opt: bool = False
