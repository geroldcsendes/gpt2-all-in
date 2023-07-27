import typing

import torch as t
from torch import Tensor
import torch.nn as nn

from gpt2.config import GPT2Config

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
        self.init_weight_scale = 1 / (config.n_layer ** 0.5)
        self.apply(self._init_weights)

        
    def forward(self, input_ids: Tensor) -> Tensor:
        pos_ids = t.arange(input_ids.size(-1), dtype=t.long, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)  # [batch_size, seq_len, n_embd]

        for block in self.blocks:
            x = block(x)
        
        x = self.unembed(self.ln(x))

        return x
    
    def _count_params(self) -> int:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")
        
        return num_params
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)): 
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        if isinstance(module, nn.Linear):
            with t.no_grad():
                module.weight *= self.init_weight_scale


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        
        self.config = config
        self.d_head = int(config.d_model / config.n_head)

        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)

        self.W_O = nn.Linear(self.d_head, config.d_model)
        self.register_buffer("mask", t.tril(t.ones(config.n_ctx, config.n_ctx)))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        
        batch_size, seq_len, d_model = x.size()

        # [batch_size, seq_len, d_model]
        Q = self.W_Q(x).view((batch_size, seq_len, self.config.n_head, self.d_head))
        K = self.W_K(x).view((batch_size, seq_len, self.config.n_head, self.d_head))    
        V = self.W_V(x).view((batch_size, seq_len, self.config.n_head, self.d_head))    

        Q = Q.permute(0, 2, 1, 3).contiguous()
        K = K.permute(0, 2, 3, 1).contiguous()
        V = V.permute(0, 2, 1, 3).contiguous()            

        QK = Q @ K / t.sqrt(t.tensor(self.d_head))  # [batch_size, n_head, seq_len, seq_len]
        QK.masked_fill_(self.mask == 0, -1e5)
        
        A = t.softmax(QK, dim=-1)

        Z = A @ V  # [batch_size, n_head, seq_len, d_head]

        Z = t.sum(self.W_O(Z), dim=1)  # [batch_size, seq_len, d_model]

        Z = self.attn_dropout(Z)

        return Z
        

class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_mlp)
        self.fc2 = nn.Linear(config.d_mlp, config.d_model)
        self.act = getattr(t.nn, config.mlp_activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x



class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.resid_dropout(self.mlp(self.ln2(x)))

        x = x + self.mlp(self.ln2(x))
        x = x + self.resid_dropout(self.mlp(self.ln2(x)))

        return x