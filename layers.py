from math import gcd
from typing import Literal

import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim,
        k_dim,
        hidden_dim,
        num_heads,
        dropout,
    ):
        super().__init__()

        self.num_heads = num_heads

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=q_dim,
            num_heads=gcd(num_heads, q_dim),
            dropout=dropout,
            kdim=k_dim,
            vdim=k_dim,
            batch_first=True,
        )

        self.ffn = nn.Linear(q_dim, hidden_dim)

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=gcd(num_heads, hidden_dim),
            dropout=dropout,
            batch_first=True,
        )

        # Normalization and activation layers
        self.cross_norm = nn.LayerNorm(q_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.activation = torch.nn.functional.silu
        self.self_norm = nn.LayerNorm(hidden_dim)

    def _create_nested_attention_mask(
        self, shapes, num_heads, max_len, num_keys, is_self_attention=False
    ):
        """Create attention masks for cross-attention or self-attention."""
        attn_mask = torch.ones(
            (len(shapes), num_heads, max_len, num_keys), dtype=torch.bool
        )
        device = next(self.parameters()).device
        for i, s in enumerate(shapes):
            for j in range(num_heads):
                for k in range(s[0]):
                    for l in range(num_keys if not is_self_attention else s[0]):
                        attn_mask[i, j, k, l] = False

        return attn_mask.view(len(shapes) * num_heads, max_len, num_keys).to(device)

    def forward(self, queries, keys, attn_mask=None, key_padding_mask=None):
        # Ensure both are 2d or 3d tensors
        assert queries.dim() == keys.dim()

        # Deal with Nested tensors
        is_nested = queries.is_nested
        if is_nested:
            # Create attention mask
            shapes = [t.shape for t in queries.unbind()]
            N = len(shapes)  # batch size
            num_heads = self.self_attention.num_heads
            max_len = max([s[0] for s in shapes])  # max sequence length
            num_keys = keys.shape[1]

            attn_mask = self._create_nested_attention_mask(
                shapes, num_heads, max_len, num_keys, is_self_attention=False
            )
            # Pad the queries
            queries = torch.nested.to_padded_tensor(queries, padding=0.0)

        # Cross-attention
        cross_output, weights = self.cross_attention(
            queries, keys, keys, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        queries = self.cross_norm(self.activation(cross_output + queries))

        # Feed-forward network
        ffn_output = self.ffn(queries)
        if queries.shape == ffn_output.shape:
            queries = self.ffn_norm(self.activation(queries + ffn_output))
        else:
            queries = self.ffn_norm(self.activation(ffn_output))

        if is_nested:
            # make mask for self-attention
            attn_mask = self._create_nested_attention_mask(
                shapes, num_heads, max_len, max_len, is_self_attention=True
            )
            attn_mask = attn_mask.view(N * num_heads, max_len, max_len).to(keys.device)
            # replace nans with zeros before self-attention, prev attention layer padded with nans
            queries = torch.nan_to_num(queries, nan=0.0)

        # Self-attention
        self_output, _ = self.self_attention(
            queries, queries, queries, attn_mask=attn_mask
        )
        queries = self.self_norm(self.activation(queries + self_output))

        # Nest again if nested since self-attention supports nested tensors
        if is_nested:
            queries = [queries[i, : s[0]] for i, s in enumerate(shapes)]
            queries = torch.nested.nested_tensor(queries)

        return queries, weights


class AttentionNetwork(nn.Module):
    def __init__(
        self,
        q_dim,
        k_dim,
        hidden_dim,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_attention_blocks: int = 1,
        normalize_inputs: bool = True,
        initial_proj: bool = True,
        proj_type: Literal["linear", "random"] = "random",
    ):
        super().__init__()
        self.normalize_inputs = normalize_inputs

        if normalize_inputs:
            self.query_norm = nn.LayerNorm(q_dim)
            self.key_norm = nn.LayerNorm(k_dim)

        # Initial projection layers
        self.proj = initial_proj
        if initial_proj:
            if proj_type == "linear":
                self.q_proj = nn.Sequential(
                    nn.Linear(q_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                )

                self.k_proj = nn.Sequential(
                    nn.Linear(k_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                )
                q_dim = hidden_dim
                k_dim = hidden_dim
            else:
                # Normalize the random projection matrix
                rand_proj_matrix = torch.randn(q_dim, hidden_dim)
                rand_proj_matrix -= rand_proj_matrix.mean(dim=0)
                rand_proj_matrix /= rand_proj_matrix.norm(dim=0)
                self.q_proj = nn.Linear(q_dim, hidden_dim, bias=False)
                self.q_proj.weight.data = rand_proj_matrix.T
                self.q_proj.weight.requires_grad = False
                self.k_proj = nn.Identity()

                q_dim = hidden_dim

        self.blocks = nn.ModuleList()
        for i in range(num_attention_blocks):
            self.blocks.append(
                AttentionBlock(
                    q_dim=q_dim if i == 0 else hidden_dim,
                    k_dim=k_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, queries, keys, attn_mask=None, key_padding_mask=None):
        if self.normalize_inputs:
            queries = self.query_norm(queries)
            keys = self.key_norm(keys)

        if hasattr(self, "q_proj"):
            queries = self.q_proj(queries)

        if hasattr(self, "k_proj"):
            keys = self.k_proj(keys)

        for block in self.blocks:
            queries, _ = block(
                queries, keys, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )

        queries = self.linear(queries).squeeze(-1)

        return queries  # , weights


def pool_attention_network(x: torch.Tensor):
    """Mean pooling over last dimension, checks if input is nested"""
    if x.is_nested:
        return torch.stack([t.mean(-1) for t in x])
    return x.mean(-1)
