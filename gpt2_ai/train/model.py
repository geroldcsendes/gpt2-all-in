from typing import Tuple

import torch as t
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
import torch.nn as nn

from gpt2_ai.train.config import GPT2Config, TrainerConfig


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config, config_trainer: TrainerConfig):
        super().__init__()
        self.config = config
        self.config_trainer = config_trainer

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.pos_embedding = nn.Embedding(
            num_embeddings=config.n_ctx, embedding_dim=config.d_model)

        self.dropout = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(
            *[Block(config, config_trainer) for _ in range(config.n_layer)])

        self.ln = nn.LayerNorm(config.d_model)

        self.unembed = nn.Linear(config.d_model, config.vocab_size)

        # init
        self.apply(self._init_weights)
        # residual projections initialized uniquely
        std = self.config.initializer_range / (self.config.n_layer ** 0.5)
        for pn, p in self.named_parameters():
            if pn.endswith('WO'):
                t.nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, input_ids: Tensor) -> Tensor:
        pos_ids = t.arange(input_ids.size(-1), dtype=t.long, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)  # [batch_size, seq_len, n_embd]

        for block in self.blocks:
            x = block(x)

        x = self.unembed(self.ln(x))

        return x

    def count_params(self, non_embed=True) -> int:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embed:
            num_params -= (self.embedding.weight.numel() + self.pos_embedding.weight.numel())

        print(f"Number of trainable parameters: {num_params:,}")

        # return num_params
        # n_params = sum(p.numel() for p in self.parameters())
        # if non_embed:
        #     n_params -= self.pos_embedding.weight.numel()
        # return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config, trainer_config: TrainerConfig):
        super().__init__()

        self.config = config
        self.d_head = int(config.d_model / config.n_head)

        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)

        self.W_O = nn.Linear(config.d_model, config.d_model)

        # only initialize causal masking if not using attention_opt
        if not trainer_config.attention_opt:
            self.register_buffer("mask", t.tril(t.ones(config.n_ctx, config.n_ctx)))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.trainer_config = trainer_config

    def forward(self, x: Tensor) -> Tensor:

        B, T, E = x.size()
        dh = self.d_head
        nh = self.config.n_head

        # [B, T, E] -> [B, nh, T, dh]
        Q = self.W_Q(x).view((B, T, nh, dh)).transpose(1, 2).contiguous()
        K = self.W_K(x).view((B, T, nh, dh)).transpose(1, 2).contiguous()
        V = self.W_V(x).view((B, T, nh, dh)).transpose(1, 2).contiguous()

        # Optionally use the context manager to ensure one of the fused kerenels is run
        # TODO handle this in a more elegeant way. Not all GPUs support all optimized
        # implementation. Let pytorch handle this automatically.
        if self.trainer_config.attention_opt:
            pdrop = self.config.attn_pdrop if self.training else 0.0
            Z = F.scaled_dot_product_attention(
                query=Q, key=K, value=V, is_causal=True, dropout_p=pdrop)

        else:
            # [B, nh, T, dh] x [B, nh, dh, T] -> [B, nh, T, T]
            QK = (Q @ K.transpose(-2, -1)) / t.sqrt(t.tensor(self.d_head))
            QK.masked_fill_(self.mask == 0, -1e5)

            A = t.softmax(QK, dim=-1)
            A = self.attn_dropout(A)

            # [[B, nh, T, T] x [B, nh, T, d_head]] -> [B, nh, T, dh]
            Z = A @ V
            # re-assemble all head outputs side by side

        Z = Z.transpose(1, 2).contiguous().view(B, T, E)
        # project back to residual size
        Z = self.W_O(Z)  # [batch_size, seq_len, d_model]

        Z = self.resid_dropout(Z)

        return Z


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_mlp)
        self.fc2 = nn.Linear(config.d_mlp, config.d_model)
        self.act = getattr(t.nn, config.mlp_activation)()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config, trainer_config: TrainerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config, trainer_config)
        self.mlp = MLP(config)
        self.config = config
        self.config_train = trainer_config

        # create dummy tensor to circumvent the issue describe in: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/16
        if self.config_train.gradient_checkpoint:
            self.dummy_tensor = t.ones(1, dtype=t.float32, requires_grad=True)


    def _add_resid(self, pre_resid_out: Tuple[Tensor, Tensor],
                   dummy_arg=Tensor) -> Tensor:
        """
        This is needed for gradient checkpointing.
        """
        pre_resid, out = pre_resid_out
        return pre_resid + out

    def forward(self, x: Tensor):

        attn_out = self.attn(self.ln1(x))

        if self.config_train.gradient_checkpoint:
            x = checkpoint.checkpoint(self._add_resid, (x, attn_out), self.dummy_tensor)
        else:
            x = x + attn_out

        mlp_out = self.mlp(self.ln2(x))

        if self.config_train.gradient_checkpoint:
            x = checkpoint.checkpoint(self._add_resid, (x, mlp_out), self.dummy_tensor)
        else:
            x = x + mlp_out

        return x