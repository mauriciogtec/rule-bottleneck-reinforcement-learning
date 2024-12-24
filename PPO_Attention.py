import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


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
        # Track whether inputs have batch dimension
        single_query = False
        single_key = False

        if query.dim() == 2:
            query = query.unsqueeze(0)  # Add batch dimension
            single_query = True
        if key.dim() == 2:
            key = key.unsqueeze(0)  # Add batch dimension
            single_key = True

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

        if attn_mask is not None:
            # Adjust binary mask (0 for masked positions, 1 for valid positions) to additive mask
            additive_mask = (1.0 - attn_mask) * -1e9
            attn_weights += additive_mask

        # Aggregate the attention weights across heads
        attn_weights = attn_weights.mean(dim=1)  # Aggregate across the head dimension

        # Remove batch dimension if it was added
        if single_query and single_key:
            attn_weights = attn_weights.squeeze(0)

        return attn_weights


# Define the Attention-based Actor Network with Multi-Head Attention
class AttentionActor(nn.Module):
    def __init__(self, state_dim, rule_dim, hidden_dim, num_heads=8):
        super(AttentionActor, self).__init__()
        # self.query_proj = nn.Linear(state_dim, hidden_dim)
        # self.key_proj = nn.Linear(rule_dim, hidden_dim)
        # self.values_proj = nn.Linear(rule_dim, hidden_dim)
        self.multihead_attn = MultiHeadAttentionWeightsOnly(
            q_dim=state_dim,
            k_dim=rule_dim,
            proj_dim=hidden_dim,
            num_heads=num_heads,
        )
        # self.actor_head = nn.Linear(hidden_dim, num_rules)  # Outputs logits over rules

    def forward(self, query, keys):
        # Project query and keys
        # query_proj = self.query_proj(query)  # Shape: [batch_size, 1, hidden_dim]
        # keys_proj = self.key_proj(keys)  # Shape: [batch_size, num_rules, hidden_dim]

        # Multi-Head Attention
        attn_logits = self.multihead_attn(query, keys)

        return attn_logits


# Define the Attention-based Critic Network with Multi-Head Attention
class AttentionCritic(nn.Module):
    def __init__(self, state_dim, rule_dim, hidden_dim, num_heads=8):
        super(AttentionCritic, self).__init__()
        # self.query_proj = nn.Linear(state_dim, hidden_dim)
        # self.key_proj = nn.Linear(rule_dim, hidden_dim)
        self.multihead_attn = MultiHeadAttentionWeightsOnly(
            q_dim=state_dim,
            k_dim=rule_dim,
            proj_dim=hidden_dim,
            num_heads=num_heads,
        )
        # self.critic_head = nn.Linear(hidden_dim, 1)  # Outputs state value

    def forward(self, query, keys):
        # Project query and keys
        # query_proj = self.query_proj(query).unsqueeze(
        #     1
        # )  # Shape: [batch_size, 1, hidden_dim]
        # keys_proj = self.key_proj(keys)  # Shape: [batch_size, num_rules, hidden_dim]

        # Multi-Head Attention
        # Multi-Head Attention
        values = self.multihead_attn(query, keys)  # Shape: [batch_size, num_rules, 1[]

        # Critic: Estimate state value
        # state_value = self.critic_head(attn_output).squeeze(-1)  # Shape: [batch_size]

        return values


# PPO Components with Separate Actor-Critic
class PPOAgent:
    def __init__(
        self,
        state_dim,
        rule_dim,
        hidden_dim,
        num_rules,
        lr=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
    ):
        self.actor = AttentionActor(state_dim, rule_dim, hidden_dim)
        self.critic = AttentionCritic(
            state_dim,
            rule_dim,
            hidden_dim,
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.trajectory = []

    def select_rules(self, query, keys):
        log_probs = self.actor(query, keys).squeeze(-2)
        values = self.critic(query, keys).squeeze(-2)
        if len(values.shape) > 1:
            values = values.squeeze(-1)
        probs = log_probs.exp()
        dist = Categorical(probs)
        action = dist.sample()
        return action, log_probs, dist.entropy(), values

    def store_transition(self, transition):
        self.trajectory.append(transition)

    def compute_advantages(self, rewards, values, dones):
        advantages, returns = [], []
        R = 0
        A = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = 0
                A = 0
            R = rewards[i] + self.gamma * R
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            A = delta + self.gamma * 0.95 * A * (1 - dones[i])
            returns.insert(0, R)
            advantages.insert(0, A)
        return torch.tensor(advantages), torch.tensor(returns)

    def update_policy(self):
        # Prepare trajectory data
        queries, keys, actions, log_probs, values, rewards, dones = zip(
            *self.trajectory
        )
        queries = torch.stack(queries)
        keys = torch.stack(keys)
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(log_probs)
        values = torch.tensor(values)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        advantages, returns = self.compute_advantages(rewards, values, dones)

        # PPO update loop
        for _ in range(10):  # Iterate over the data multiple times
            # Actor update
            logits = self.actor(queries, keys)
            dist = Categorical(logits=logits)

            new_log_probs = F.log_softmax(logits, dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages
            )
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # Critic update
            new_values = self.critic(queries, keys).squeeze(-1)
            value_loss = (returns - new_values).pow(2).mean()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        # Clear trajectory
        self.trajectory = []


# Define Environment
class Environment:
    def __init__(self, state_dim, rule_dim, num_rules):
        self.state_dim = state_dim
        self.rule_dim = rule_dim
        self.num_rules = num_rules
        self.reset()

    def reset(self):
        self.state = torch.randn((1, self.state_dim))  # Random initial state
        self.rules = torch.randn((1, self.num_rules, self.rule_dim))  # Random rules
        return self.state, self.rules

    def step(self, action):
        # Define how the state and reward change given an action
        reward = torch.rand(1).item()  # Random reward
        done = (
            torch.rand(1).item() > 0.95
        )  # Randomly end the episode with 5% probability
        self.state = torch.randn((1, self.state_dim))  # New random state
        return self.state, reward, done


# Example Usage
if __name__ == "__main__":
    # # Example dimensions (adjust as needed)
    # state_dim = 128  # Dimensionality of query embeddings
    # rule_dim = 128  # Dimensionality of rule embeddings
    # hidden_dim = 64
    # num_rules = 10

    # # Initialize PPO agent and environment
    # agent = PPOAgent(state_dim, rule_dim, hidden_dim, num_rules)
    # env = Environment(state_dim, rule_dim, num_rules)

    # # Training loop
    # for episode in range(100):
    #     state, rules = env.reset()
    #     done = False
    #     values = []
    #     while not done:
    #         action, log_prob, entropy, value = agent.select_action(state, rules)
    #         next_state, reward, done = env.step(action.item())
    #         values.append(value.item())
    #         agent.store_transition((state, rules, action, log_prob, value, reward, done))
    #         state = next_state

    #     # Append final value for advantage calculation
    #     agent.store_transition((None, None, None, None, torch.tensor(0.0), 0, True))
    #     agent.update_policy()
    #     print(f"Episode {episode + 1} complete")
    from src.agent import call_for_action, gen_rules
    from Embedding_rule_seq import (
        generate_rule_combinations,
        generate_embeddings_for_rules,
    )
    from src.language_wrappers import HeatAlertsWrapper
    from weather2alert.env import HeatAlertEnv
    from langchain_together import TogetherEmbeddings, ChatTogether

    embed_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    chat_model = ChatTogether(model="meta-llama/Llama-3.2-3B-Instruct-Turbo")
    env = HeatAlertsWrapper(HeatAlertEnv(), embed_model)
    action_state_text = env.action_space_text
    task_text = env.task_text

    # Example dimensions (adjust as needed)
    state_dim = 768  # Dimensionality of query embeddings
    rule_dim = 768  # Dimensionality of rule embeddings
    hidden_dim = 768
    num_rules = 10
    agent = PPOAgent(state_dim, rule_dim, hidden_dim, num_rules)

    # print number of trainable parameters
    print(
        f"Number of trainable parameters in actor: {sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)}"
    )

    # Training loop
    for episode in range(100):
        obs, info = env.reset()
        state_emb = obs
        done = False
        step = 0
        values = []

        while not done:
            state_text = info["obs_text"]
            rules_text = gen_rules(
                chat_model,
                state_text,
                action_state_text,
                task_text,
                num_rules,
                verbose=False,
            )
            print(f"Generated {len(rules_text)} rules:")
            rules_text = generate_rule_combinations(rules_text, max_combinations=2)
            rules_emb = generate_embeddings_for_rules(rules_text, embed_model)

            # Convert embeddings to tensors
            state_emb = torch.tensor(state_emb, dtype=torch.float32).unsqueeze(0)
            rules_emb = torch.tensor(rules_emb, dtype=torch.float32)

            # Here the action is the internal action (i.e.,) the rules
            # Rules = Internal actions
            with torch.no_grad():
                sel_rule_index, log_prob, entropy, rule_values = agent.select_rules(
                    state_emb, rules_emb
                )
            sel_rule_text = rules_text[sel_rule_index.item()]
            sel_rule_value = rule_values[sel_rule_index.item()]

            values.append(sel_rule_value.item())

            # Get external env action
            env_action, expl = call_for_action(
                chat_model,
                state_text,
                sel_rule_text,
                action_state_text,
                task_text,
                verbose=False,
            )

            # Step environment
            obs, reward, terminated, truncated, info = env.step(env_action)
            state_emb = obs

            # Store transition
            agent.store_transition(
                (
                    state_emb,
                    rules_emb,
                    sel_rule_index,
                    log_prob,
                    sel_rule_value,
                    reward,
                    done,
                )
            )
            step += 1

            # Print info: step, state, selected rule, reward, done
            print(f"Step: {step}")
            print(f"State: {state_text}")
            print(f"Selected Rule: {sel_rule_text}")
            print(f"External Action: {env_action}")
            print(f"Explanation: {expl}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")

        # Append final value for advantage calculation
        agent.store_transition((None, None, None, None, torch.tensor(0.0), 0, True))
        agent.update_policy()
        print(f"Episode {episode + 1} complete")
