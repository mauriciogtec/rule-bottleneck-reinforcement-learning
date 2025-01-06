import torch
from torch import nn


class MultiHeadAttentionWeightsOnly(nn.Module):
    def __init__(self, q_dim, k_dim, num_heads, embed_dim, dropout=0.0):
        super(MultiHeadAttentionWeightsOnly, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Projection dimension must be divisible by the number of heads."

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.num_heads = num_heads
        self.proj_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        # Linear layers for query and key transformations
        self.q_proj = nn.Linear(q_dim, embed_dim)
        self.k_proj = nn.Linear(k_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, querys, keys, attn_mask=None, key_padding_mask=None):
        # Input shapes:
        #   query: (batch_size, query_len, q_dim)
        #   key: (batch_size, key_len, k_dim)
        #   attn_mask: (batch_size, key_len, key_len)
        # Output shape:
        #   attn_weights: (batch_size, query_len, key_len)

        # check key and query have the same batch dimension
        assert (
            keys.dim() == querys.dim()
        ), "Query and key must have the same batch dimension."

        # check dimensions is 2D or 3D
        assert keys.dim() in [2, 3], "Input key must be 2D or 3D tensor."

        # Add batch dimension if it doesn't exist
        is_batched = keys.dim() == 3
        if not is_batched:
            querys = querys.unsqueeze(0)
            keys = keys.unsqueeze(0)

        batch_size, query_len, q_dim = querys.size()
        _, key_len, k_dim = keys.size()

        assert q_dim == self.q_dim, "Query dimension must match the initialized q_dim."
        assert k_dim == self.k_dim, "Key dimension must match the initialized k_dim."

        # Linear transformations for query and key
        Q = self.q_proj(querys)  # (batch_size, query_len, proj_dim)
        K = self.k_proj(keys)  # (batch_size, key_len, proj_dim)

        # Apply dropout after projections
        Q = self.dropout(Q)
        K = self.dropout(K)

        # Reshape and transpose for multi-head computation
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention weights (logits)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        # attn_weights shape: (batch_size, num_heads, query_len, key_len)

        # Aggregate the attention weights across heads
        attn_weights = attn_weights.mean(dim=1)  # Aggregate across the head dimension

        # merge key padding and attention mask
        mask = torch.ones(batch_size, query_len, key_len).to(attn_weights.device)

        if key_padding_mask is not None:
            mask *= key_padding_mask.unsqueeze(1).float()

        if attn_mask is not None:
            mask *= attn_mask.float()

        # Apply the mask
        attn_weights = attn_weights + (1.0 - mask) * -1e9

        # Remove batch dimension if it was added for non-batch inputs
        if not is_batched:
            attn_weights = attn_weights.squeeze(0)

        return attn_weights


# Define the Attention-based Actor Network with Multi-Head Attention
class AttentionActorCritic(nn.Module):
    def __init__(
        self,
        q_dim,
        k_dim,
        hidden_dim,
        num_heads: int = 8,
        dropout: float = 0.0,
        q_proj: bool = True,
        k_proj: bool = True,
        num_attention_layers: int = 2,
        num_critic_layers: int = 2,
    ):
        super().__init__()
        assert (
            num_attention_layers >= 1
        ), "Number of attention layers must be at least 1."
        assert num_critic_layers >= 1, "Number of critic layers must be at least 1."

        if q_proj:
            self.q_proj = nn.Sequential(nn.Linear(q_dim, hidden_dim), nn.SiLU())
            self.q_dim = hidden_dim
        else:
            self.q_dim = q_dim

        if k_proj:
            self.k_proj = nn.Sequential(nn.Linear(k_dim, hidden_dim), nn.SiLU())
            self.k_dim = hidden_dim
        else:
            self.k_dim = k_dim

        # cross-attention full multi-head attention
        if num_attention_layers == 1:
            self.attn_body = []
        else:
            self.attn_body = nn.ModuleList()
            for i in range(num_attention_layers - 1):
                embed_dim = self.q_dim if i == 0 else hidden_dim
                self.attn_body.append(
                    nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        kdim=self.k_dim,
                        vdim=hidden_dim,
                        batch_first=True,
                    )
                )

        # multi-head attention with weights only
        self.attn_head = MultiHeadAttentionWeightsOnly(
            q_dim=self.q_dim,
            k_dim=self.k_dim,
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # critic network
        self.critic = nn.Sequential()
        for i in range(num_critic_layers):
            in_dim = self.q_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_critic_layers - 1 else 1
            self.critic.append(nn.Linear(in_dim, out_dim))
            if i < num_critic_layers - 1:
                self.critic.append(nn.SiLU())

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query, keys, attn_mask=None, key_padding_mask=None):
        # first project inputs if needed
        if hasattr(self, "q_proj"):
            query = self.q_proj(query)

        if hasattr(self, "k_proj"):
            keys = self.k_proj(keys)

        # get value from critic network
        value = self.critic(query).squeeze(-1)

        # loop through body attention layers
        for attn_layer in self.attn_body:
            query, _ = attn_layer(
                query,
                keys,
                keys,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            query = torch.nn.functional.silu(query)

        # final attention layer
        attn_logits = self.attn_head(
            query, keys, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        return attn_logits, value


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
