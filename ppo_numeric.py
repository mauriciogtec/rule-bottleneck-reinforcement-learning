# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import envs as E
import layers


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rulebots"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    log_interval: int = 1
    """the log interval"""

    # Algorithm specific arguments
    env_id: str = "UgandaNumeric"
    """the id of the environment"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 64
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # architcture
    hidden_dim: int = 16
    """the hidden dimension"""

    # eval
    num_eval_episodes: int = 16
    """the number of steps to run in each eval environment per policy rollout"""
    eval_interval: int = 1
    """the evaluation interval"""
    eval_deterministic: bool = True
    """if toggled, the evaluation will be deterministic"""
    max_episode_steps: int = 64
    """the maximum number of steps per episode"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    agent: Literal["ppo_numeric", "ppo_random_test"] = "ppo_numeric"
    """the agent to evaluate"""  # used for logging


def make_env(env_id, seed, max_episode_steps=None):
    def scale_reward(r):
        return r / max_episode_steps

    def thunk():
        env = gym.make(env_id)
        if env_id == "HeatAlertsNumeric":
            env = gym.wrappers.TransformReward(env, func=scale_reward)
        elif env_id == "UgandaNumeric":
            env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"numeric_ppo_eval__{args.env_id}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    train_env_funs = [
        make_env(args.env_id, args.seed + i, args.max_episode_steps)
        for i in range(args.num_envs)
    ]
    eval_env_funs = [
        make_env(args.env_id, 1000 * args.seed + i, args.max_episode_steps)
        for i in range(args.num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(train_env_funs)
    eval_envs = gym.vector.SyncVectorEnv(eval_env_funs)

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # actor_network = layers.SelfAttentionNetwork(
    #     q_dim=envs.single_observation_space.shape[-1],
    #     output_dim=envs.single_action_space.n,
    #     hidden_dim=args.hidden_dim,
    #     normalize_inputs=False,
    #     num_heads=1,
    # )

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    actor_network = nn.Sequential(
        nn.LayerNorm(envs.single_observation_space.shape[-1]),
        layer_init(nn.Linear(envs.single_observation_space.shape[-1], args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), std=1.0),
    )

    value_network = nn.Sequential(
        nn.LayerNorm(envs.single_observation_space.shape[-1]),
        layer_init(nn.Linear(envs.single_observation_space.shape[-1], args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
        nn.SiLU(),
        nn.LayerNorm(args.hidden_dim),
        layer_init(nn.Linear(args.hidden_dim, 1), 0.01),
    )
    agent = nn.ModuleDict({"actor": actor_network, "critic": value_network})
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    print("==== Actor ====")
    torchsummary.summary(actor_network)
    print("==== Critic ====")
    torchsummary.summary(value_network)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    autoreset = np.zeros(args.num_envs, dtype=bool)
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_logits = actor_network(obs[step])
                action_dist = Categorical(logits=action_logits)
                if args.agent == "ppo_random_test":
                    action = envs.action_space.sample()
                    action = torch.tensor(action).to(device)
                else:
                    action = action_dist.sample()
                logprob = action_dist.log_prob(action)
                value = value_network(obs[step])
                # action, logprob, _, value = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )

            if "episode" in infos:
                for i in range(args.num_envs):
                    if infos["_episode"][i]:
                        r, l = infos["episode"]["r"][i], infos["episode"]["l"][i]
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)

                        print(f"global_step={global_step}, episodic_return={r:.4f}")

            autoreset = np.logical_or(autoreset, terminations).any()
            if autoreset.any():
                autoreset = np.zeros(args.num_envs, dtype=bool)
                # fake action to reset
                # assumes autoreset all at same time

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action_logits = actor_network(next_obs)
                    action_dist = Categorical(logits=action_logits)
                    action = action_dist.sample()
                    logprob = action_dist.log_prob(action)
                    value = value_network(next_obs)
                    # action, logprob, _, value = agent.get_action_and_value(next_obs)

                next_obs, reward, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )

            next_done = terminations
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            rewards[step] = torch.tensor(reward).to(device).view(-1)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = value_network(next_obs).reshape(-1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                newlogits = actor_network(b_obs[mb_inds])
                newdist = Categorical(logits=newlogits)
                newlogprob_all = newlogits.log_softmax(-1)
                newlogprob = newlogprob_all.gather(
                    1, b_actions[mb_inds].long().view(-1, 1)
                ).view(-1)
                entropy = newdist.entropy().mean()
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                newvalue = value_network(b_obs[mb_inds]).view(-1)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if iteration % args.log_interval == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

        # Run eval episodes
        if iteration % args.eval_interval == 0:
            eval_returns = []
            eval_obs, _ = eval_envs.reset(seed=args.seed)
            eval_obs = torch.Tensor(eval_obs).to(device)
            eval_done = torch.zeros(args.num_envs).to(device)
            num_eval_episodes = 0

            while num_eval_episodes < args.num_eval_episodes:
                for _ in range(args.num_eval_episodes):
                    with torch.no_grad():
                        action_logits = actor_network(eval_obs)
                        action_dist = Categorical(logits=action_logits)
                        action = action_dist.probs.argmax(dim=-1)
                eval_obs, reward, _, _, infos = eval_envs.step(action.cpu().numpy())
                eval_obs = torch.Tensor(eval_obs).to(device).to(device)

                if "episode" in infos:
                    for i in range(args.num_envs):
                        if infos["_episode"][i]:
                            r, l = infos["episode"]["r"][i], infos["episode"]["l"][i]

                            eval_returns.append(r)
                            num_eval_episodes += 1

            mean_eval_returns = np.mean(eval_returns)
            writer.add_scalar("charts/eval_return", mean_eval_returns, global_step)
            print(f"global_step={global_step}, eval_return={mean_eval_returns:.4f}")

    envs.close()
    writer.close()
