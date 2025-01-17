from math import gcd

# from typing import Literal, Optional
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
            batch_first=True,
        )

        self.ffn = nn.Linear(q_dim, hidden_dim)

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=gcd(num_heads, hidden_dim),
            dropout=dropout,
            kdim=hidden_dim,
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
    ):
        super().__init__()
        self.normalize_inputs = normalize_inputs

        if normalize_inputs:
            self.query_norm = nn.LayerNorm(q_dim)
            self.key_norm = nn.LayerNorm(k_dim)

        # Initial projection layers
        self.proj = initial_proj
        if initial_proj:
            self.q_proj = nn.Sequential(
                nn.Linear(q_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            q_dim = hidden_dim

            self.k_proj = nn.Sequential(
                nn.Linear(k_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            k_dim = hidden_dim

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


class AttentionActorCritic(nn.Module):
    """Actor where the output is the action logits and the critic is a state-only value function."""

    def __init__(
        self,
        q_dim,
        k_dim,
        hidden_dim,
        num_heads: int = 8,
        dropout: float = 0.0,
        proj: bool = True,
        num_attention_layers: int = 2,
        num_critic_layers: int = 2,
        normalize_inputs: bool = True,  # Input normalization flag
        sequential_critic: bool = False,
    ):
        super().__init__()
        assert (
            num_attention_layers >= 1
        ), "Number of attention layers must be at least 1."
        assert num_critic_layers >= 1, "Number of critic layers must be at least 1."

        self.actor = AttentionNetwork(
            q_dim=q_dim,
            k_dim=k_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_attention_blocks=num_attention_layers,
            normalize_inputs=normalize_inputs,
            proj=proj,
            weights_only=True,
        )

        # state only critic, output dim should be 1
        self.sequential_critic = sequential_critic
        if not sequential_critic:
            self.critic = AttentionNetwork(
                q_dim=q_dim,
                k_dim=k_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_attention_blocks=num_critic_layers,
                normalize_inputs=normalize_inputs,
                output_dim=1,
                proj=proj,
            )
        else:
            self.critic = nn.Sequential()
            for i in range(num_critic_layers):
                in_dim = self.actor.q_dim if i == 0 else hidden_dim
                out_dim = hidden_dim if i < num_critic_layers - 1 else 1
                self.critic.append(nn.Linear(in_dim, out_dim))
                if i < num_critic_layers - 1:
                    self.critic.append(nn.SiLU())
                    self.critic.append(nn.LayerNorm(out_dim))

    def forward(self, queries, keys, attn_mask=None, key_padding_mask=None):
        attn_logits = self.actor(queries, keys, attn_mask, key_padding_mask)

        if self.sequential_critic:
            value = self.critic(queries)
        else:
            value = self.critic(queries, keys, attn_mask, key_padding_mask)
        value = value.squeeze(-1)

        return attn_logits, value


# # Define the Attention-based Actor Network with Multi-Head Attention
# class AttentionActorCritic(nn.Module):
#     def __init__(
#         self,
#         q_dim,
#         k_dim,
#         hidden_dim,
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         q_proj: bool = True,
#         k_proj: bool = True,
#         num_attention_layers: int = 2,
#         num_critic_layers: int = 2,
#         normalize_inputs: bool = True,  # Input normalization flag
#     ):
#         super().__init__()
#         assert (
#             num_attention_layers >= 1
#         ), "Number of attention layers must be at least 1."
#         assert num_critic_layers >= 1, "Number of critic layers must be at least 1."

#         self.actor = _build_attention_network(
#             q_dim,
#             k_dim,
#             hidden_dim,
#             num_heads,
#             dropout,
#             q_proj,
#             k_proj,
#             num_attention_layers,
#             normalize_inputs,
#         )

#         # state only state
#         self.critic = nn.Sequential()
#         for i in range(num_critic_layers):
#             in_dim = self.actor.q_dim if i == 0 else hidden_dim
#             out_dim = hidden_dim if i < num_critic_layers - 1 else 1
#             self.critic.append(nn.Linear(in_dim, out_dim))
#             if i < num_critic_layers - 1:
#                 self.critic.append(nn.SiLU())
#                 self.critic.append(nn.LayerNorm(out_dim))

#         for m in self.modules():
#             if hasattr(m, "weight"):
#                 nn.init.kaiming_normal_(m.weight)
#             if hasattr(m, "bias") and m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     def forward(self, query, keys, attn_mask=None, key_padding_mask=None):
#         if self.actor.normalize_inputs:
#             query = self.actor.query_norm(query)
#             keys = self.actor.key_norm(keys)

#         if hasattr(self.actor, "q_proj"):
#             query = self.actor.q_proj(query)

#         if hasattr(self.actor, "k_proj"):
#             keys = self.actor.k_proj(keys)

#         value = self.critic(query).squeeze(-1)

#         for i in range(0, len(self.actor.attn_body), 3):
#             attn_layer = self.actor.attn_body[i]
#             act_layer = self.actor.attn_body[i + 1]
#             norm_layer = self.actor.attn_body[i + 2]

#             layer_output, _ = attn_layer(
#                 query,
#                 keys,
#                 keys,
#                 attn_mask=attn_mask,
#                 key_padding_mask=key_padding_mask,
#             )
#             query = norm_layer(act_layer(query + layer_output))

#         attn_logits = self.actor.attn_head(
#             query, keys, attn_mask=attn_mask, key_padding_mask=key_padding_mask
#         )

#         return attn_logits, value


# # Define the Attention-based Actor Network with Multi-Head Attention
# class AttentionActor(nn.Module):
#     def __init__(
#         self,
#         q_dim,
#         k_dim,
#         hidden_dim,
#         num_heads: int = 8,
#         dropout: float = 0.0,
#         q_proj: bool = True,
#         k_proj: bool = True,
#         num_attention_layers: int = 2,
#         normalize_inputs: bool = True,  # Input normalization flag
#     ):
#         super().__init__()
#         assert (
#             num_attention_layers >= 1
#         ), "Number of attention layers must be at least 1."

#         self.normalize_inputs = normalize_inputs

#         if normalize_inputs:
#             self.query_norm = nn.LayerNorm(q_dim)
#             self.key_norm = nn.LayerNorm(k_dim)

#         if q_proj:
#             self.q_proj = nn.Sequential(
#                 nn.Linear(q_dim, hidden_dim),
#                 nn.SiLU(),
#                 nn.LayerNorm(hidden_dim),
#             )
#             self.q_dim = hidden_dim
#         else:
#             self.q_dim = q_dim

#         if k_proj:
#             self.k_proj = nn.Sequential(
#                 nn.Linear(k_dim, hidden_dim),
#                 nn.SiLU(),
#                 nn.LayerNorm(hidden_dim),
#             )
#             self.k_dim = hidden_dim
#         else:
#             self.k_dim = k_dim

#         if num_attention_layers == 1:
#             self.attn_body = []
#         else:
#             self.attn_body = nn.ModuleList()
#             for i in range(num_attention_layers - 1):
#                 embed_dim = self.q_dim if i == 0 else hidden_dim
#                 self.attn_body.append(
#                     nn.MultiheadAttention(
#                         embed_dim=embed_dim,
#                         num_heads=num_heads,
#                         dropout=dropout,
#                         kdim=self.k_dim,
#                         vdim=hidden_dim,
#                         batch_first=True,
#                         add_bias_kv=True,
#                     )
#                 )
#                 self.attn_body.append(nn.SiLU())
#                 self.attn_body.append(nn.LayerNorm(embed_dim))

#         self.attn_head = MultiHeadAttentionWeightsOnly(
#             q_dim=self.q_dim,
#             k_dim=self.k_dim,
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#         )

#         for m in self.modules():
#             if hasattr(m, "weight"):
#                 nn.init.kaiming_normal_(m.weight)
#             if hasattr(m, "bias") and m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     def forward(self, query, keys, attn_mask=None, key_padding_mask=None):
#         if self.normalize_inputs:
#             query = self.query_norm(query)
#             keys = self.key_norm(keys)

#         if hasattr(self, "q_proj"):
#             query = self.q_proj(query)

#         if hasattr(self, "k_proj"):
#             keys = self.k_proj(keys)

#         for i in range(0, len(self.attn_body), 3):
#             attn_layer = self.attn_body[i]
#             act_layer = self.attn_body[i + 1]
#             norm_layer = self.attn_body[i + 2]

#             layer_output, _ = attn_layer(
#                 query,
#                 keys,
#                 keys,
#                 attn_mask=attn_mask,
#                 key_padding_mask=key_padding_mask,
#             )
#             query = norm_layer(act_layer(query + layer_output))

#         attn_logits = self.attn_head(
#             query, keys, attn_mask=attn_mask, key_padding_mask=key_padding_mask
#         )

#         return attn_logits


if __name__ == "__main__":
    # Test the MultiHeadAttentionWeightsOnly module
    querys = torch.randn(2, 5, 8)
    keys = torch.randn(2, 7, 32)
    # atn mask must be a 2D tensor of
    attn_mask = torch.rand(5, 7).round().bool()
    key_padding_mask = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool
    )
    # Test the AttentionActorCritic module
    actor = AttentionActorCritic(
        q_dim=8,
        k_dim=32,
        hidden_dim=64,
        num_heads=8,
        dropout=0.25,
        q_proj=True,
        k_proj=True,
    )

    attn_logits, value = actor(querys, keys, attn_mask, key_padding_mask)

    print("Attention logits shape:", attn_logits.shape)
