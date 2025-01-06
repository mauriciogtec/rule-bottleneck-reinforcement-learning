import torch
from torch import nn


class MultiHeadAttentionWeightsOnly(nn.Module):
    def __init__(self, q_dim, k_dim, num_heads, proj_dim, dropout=0.0):
        super(MultiHeadAttentionWeightsOnly, self).__init__()

        assert (
            proj_dim % num_heads == 0
        ), "Projection dimension must be divisible by the number of heads."

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        self.head_dim = proj_dim // num_heads

        # Linear layers for query and key transformations
        self.q_proj = nn.Linear(q_dim, proj_dim)
        self.k_proj = nn.Linear(k_dim, proj_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, attn_mask=None):
        # Input shapes:
        #   query: (batch_size, query_len, q_dim)
        #   key: (batch_size, key_len, k_dim)
        #   attn_mask: (batch_size, query_len, key_len)
        # Output shape:
        #   attn_weights: (batch_size, query_len, key_len)

        # check key and query have the same batch dimension
        assert (
            key.dim() == query.dim()
        ), "Query and key must have the same batch dimension."

        # check dimensions is 2D or 3D
        assert key.dim() in [2, 3], "Input key must be 2D or 3D tensor."


        # Add batch dimension if it doesn't exist
        is_batched = key.dim() == 3
        if not is_batched:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)

        batch_size, query_len, q_dim = query.size()
        _, key_len, k_dim = key.size()

        assert q_dim == self.q_dim, "Query dimension must match the initialized q_dim."
        assert k_dim == self.k_dim, "Key dimension must match the initialized k_dim."

        # Linear transformations for query and key
        Q = self.q_proj(query)  # (batch_size, query_len, proj_dim)
        K = self.k_proj(key)  # (batch_size, key_len, proj_dim)

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

        if attn_mask is not None:
            # Adjust binary mask (0 for masked positions, 1 for valid positions) to additive mask
            # Here the unsqueezing is done to allow for broadcasting for the query_len dimension
            additive_mask = (1.0 - attn_mask).unsqueeze(1) * -1e9
            attn_weights += additive_mask  # TODO: debug this

        # Remove batch dimension if it was added for non-batch inputs
        if not is_batched:
            attn_weights = attn_weights.squeeze(0)

        return attn_weights


# Define the Attention-based Actor Network with Multi-Head Attention
class AttentionActor(nn.Module):
    def __init__(self, state_dim, rule_dim, hidden_dim, num_heads=8, dropout=0.0):
        super(AttentionActor, self).__init__()
        # self.query_proj = nn.Linear(state_dim, hidden_dim)
        # self.key_proj = nn.Linear(rule_dim, hidden_dim)
        # self.values_proj = nn.Linear(rule_dim, hidden_dim)
        self.multihead_attn = MultiHeadAttentionWeightsOnly(
            q_dim=state_dim,
            k_dim=rule_dim,
            proj_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        # self.actor_head = nn.Linear(hidden_dim, num_rules)  # Outputs logits over rules

    def forward(self, query, keys, attn_mask=None):
        # Project query and keys
        # query_proj = self.query_proj(query)  # Shape: [batch_size, 1, hidden_dim]
        # keys_proj = self.key_proj(keys)  # Shape: [batch_size, num_rules, hidden_dim]

        # Multi-Head Attention
        attn_logits = self.multihead_attn(query, keys, attn_mask=attn_mask)

        return attn_logits

    def sample(self, query, keys, attn_mask=None):
        attn_logits = self(query, keys, attn_mask=attn_mask)
        attn_dist = torch.distributions.Categorical(logits=attn_logits)
        rule_idx = attn_dist.sample()
        return rule_idx, attn_dist.log_prob(rule_idx)
    


# Define the Attention-based Critic Network with Multi-Head Attention
class AttentionCritic(nn.Module):
    def __init__(self, state_dim, rule_dim, hidden_dim, num_heads=1, dropout=0.0):
        super(AttentionCritic, self).__init__()
        # self.query_proj = nn.Linear(state_dim, hidden_dim)
        # self.key_proj = nn.Linear(rule_dim, hidden_dim)
        self.multihead_attn = MultiHeadAttentionWeightsOnly(
            q_dim=state_dim,
            k_dim=rule_dim,
            proj_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        # self.critic_head = nn.Linear(hidden_dim, 1)  # Outputs state value

    def forward(self, query, keys, attn_mask=None):
        # Project query and keys
        # query_proj = self.query_proj(query).unsqueeze(
        #     1
        # )  # Shape: [batch_size, 1, hidden_dim]
        # keys_proj = self.key_proj(keys)  # Shape: [batch_size, num_rules, hidden_dim]

        # Multi-Head Attention
        # Multi-Head Attention
        values = self.multihead_attn(
            query, keys, attn_mask=attn_mask
        )  # Shape: [batch_size, 1, num_rules]

        # Critic: Estimate state value
        # state_value = self.critic_head(attn_output).squeeze(-1)  # Shape: [batch_size]

        return values
