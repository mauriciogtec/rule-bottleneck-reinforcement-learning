import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Define the Attention-based Actor Network with Multi-Head Attention
class AttentionActor(nn.Module):
    def __init__(self, state_dim, rule_dim, hidden_dim, num_rules, num_heads=4):
        super(AttentionActor, self).__init__()
        self.query_proj = nn.Linear(state_dim, hidden_dim)
        self.key_proj = nn.Linear(rule_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.actor_head = nn.Linear(hidden_dim, num_rules)  # Outputs logits over rules

    def forward(self, query, keys):
        # Project query and keys
        query_proj = self.query_proj(query).unsqueeze(
            1
        )  # Shape: [batch_size, 1, hidden_dim]
        keys_proj = self.key_proj(keys)  # Shape: [batch_size, num_rules, hidden_dim]

        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(query_proj, keys_proj, keys_proj)
        attn_output = attn_output.squeeze(1)  # Shape: [batch_size, hidden_dim]

        # Actor: Convert attention output to logits
        logits = self.actor_head(attn_output)  # Shape: [batch_size, num_rules]
        probs = torch.softmax(logits, dim=-1)
        return probs


# Define the Attention-based Critic Network with Multi-Head Attention
class AttentionCritic(nn.Module):
    def __init__(self, state_dim, rule_dim, hidden_dim, num_rules, num_heads=4):
        super(AttentionCritic, self).__init__()
        self.query_proj = nn.Linear(state_dim, hidden_dim)
        self.key_proj = nn.Linear(rule_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.critic_head = nn.Linear(hidden_dim, 1)  # Outputs state value

    def forward(self, query, keys):
        # Project query and keys
        query_proj = self.query_proj(query).unsqueeze(
            1
        )  # Shape: [batch_size, 1, hidden_dim]
        keys_proj = self.key_proj(keys)  # Shape: [batch_size, num_rules, hidden_dim]

        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(query_proj, keys_proj, keys_proj)
        attn_output = attn_output.squeeze(1)  # Shape: [batch_size, hidden_dim]

        # Critic: Estimate state value
        state_value = self.critic_head(attn_output).squeeze(-1)  # Shape: [batch_size]
        return state_value


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
        self.actor = AttentionActor(state_dim, rule_dim, hidden_dim, num_rules)
        self.critic = AttentionCritic(state_dim, rule_dim, hidden_dim, num_rules)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.trajectory = []

    def select_rules(self, query, keys):
        probs = self.actor(query, keys)
        value = self.critic(query, keys)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

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
            probs = self.actor(queries, keys)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
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
    from Embedding_rule_seq import generate_rule_combinations, generate_embeddings_for_rules
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
    hidden_dim = 32
    num_rules = 10
    agent = PPOAgent(state_dim, rule_dim, hidden_dim, num_rules)

    # Training loop
    for episode in range(100):
        state_emb, info = env.reset()
        done = False
        values = []

        while not done:
            state_text = info["obs_text"]
            rules_text = gen_rules(
                chat_model,
                state_text,
                action_state_text,
                task_text,
                num_rules,
                verbose=True,
            )
            rules_text = generate_rule_combinations(rules_text)
            rules_emb = generate_embeddings_for_rules(rules_text, embed_model)

            # Here the action is the internal action (i.e.,) the rules
            # Rules = Internal actions
            sel_rule_index, log_prob, entropy, value = agent.select_rules(state_emb, rules_emb)
            sel_rule_text = rules_text[sel_rule_index.item()]
            agent.store_transition((state_emb, rules_emb, sel_rule_index, log_prob, value, reward, done))
            values.append(value.item())

            # Get external env action
            env_action = call_for_action(
                chat_model, state_text, sel_rule_text, action_state_text, task_text, verbose=True
            )

            # Step environment
            next_state, reward, done = env.step(env_action)
            state = next_state

        # Append final value for advantage calculation
        agent.store_transition((None, None, None, None, torch.tensor(0.0), 0, True))
        agent.update_policy()
        print(f"Episode {episode + 1} complete")
