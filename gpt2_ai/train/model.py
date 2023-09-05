import typing

import torch as t
import torch.utils.checkpoint as checkpoint
from torch import Tensor
import torch.nn as nn

from gpt2_ai.train.config import GPT2Config


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.pos_embedding = nn.Embedding(
            num_embeddings=config.n_ctx, embedding_dim=config.d_model)

        self.dropout = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

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
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.config = config
        self.d_head = int(config.d_model / config.n_head)

        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)

        self.W_O = nn.Linear(config.d_model, config.d_model)
        self.register_buffer("mask", t.tril(t.ones(config.n_ctx, config.n_ctx)))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:

        B, T, E = x.size()
        dh = self.d_head
        nh = self.config.n_head

        # [B, T, E] -> [B, nh, T, dh]
        Q = self.W_Q(x).view((B, T, nh, dh)).transpose(1, 2).contiguous()
        K = self.W_K(x).view((B, T, nh, dh)).transpose(1, 2).contiguous()
        V = self.W_V(x).view((B, T, nh, dh)).transpose(1, 2).contiguous()

        # [B, nh, T, dh] x [B, nh, dh, T] -> [B, nh, T, T]
        QK = (Q @ K.transpose(-2, -1)) / t.sqrt(t.tensor(self.d_head))
        QK.masked_fill_(self.mask == 0, -1e5)

        A = t.softmax(QK, dim=-1)
        A = self.attn_dropout(A)

        # [[B, nh, T, T] x [B, nh, T, d_head]] -> [B, nh, T, dh]
        Z = A @ V
        # re-assemble all head outputs side by side
        Z = Z.transpose(1, 2).contiguous().view(B, T, E)

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
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.config = config

    def forward(self, x: Tensor):

        if self.config.gradient_checkpoint:
            ln_out = self.ln1(x)
            attn_out = checkpoint.checkpoint(self.attn, ln_out)
            # attn_out = checkpoint.checkpoint(self.attn(self.ln1), x)
        else:
            attn_out = self.attn(self.ln1(x))

        x = x + attn_out

        if self.config.gradient_checkpoint:
            ln_out = self.ln2(x)
            mlp_out = checkpoint.checkpoint(self.mlp, ln_out)
        else:
            mlp_out = self.mlp(self.ln2(x))

        x = x + mlp_out

        # x = x + self.attn(self.ln1(x))
        # x = x + self.mlp(self.ln2(x))

        return x