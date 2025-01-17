## Rule-based Reinforcement Learning for More Interpretable and Efficient Language Agents in Resource-Constrained Allocation Tasks

Run the code with the following command:
```bash
python sac_attention.py --llm="gpt-4o-mini"
```

Visualize the results with TensorBoard:
```bash
tensorboard --logdir=runs
```
Or by running the code with the flag ``--track`` and visualizing in Weights & Biases.

To see the full functionality of the code, run:
```bash
python sac_attention.py --help
```

```
usage: sac_attention.py [-h] [OPTIONS]

╭─ options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                                                            │
│ --exp-name STR          the name of this experiment (default: sac_attention)                                                                       │
│ --seed INT              seed of the experiment (default: 1)                                                                                        │
│ --torch-deterministic, --no-torch-deterministic                                                                                                    │
│                         if toggled, `torch.backends.cudnn.deterministic=False` (default: True)                                                     │
│ --cuda, --no-cuda       if toggled, cuda will be enabled by default (default: True)                                                                │
│ --track, --no-track     if toggled, this experiment will be tracked with Weights and Biases (default: False)                                       │
│ --wandb-project-name STR                                                                                                                           │
│                         the wandb's project name (default: rulebots)                                                                               │
│ --wandb-entity {None}|STR                                                                                                                          │
│                         the entity (team) of wandb's project (default: None)                                                                       │
│ --capture-video, --no-capture-video                                                                                                                │
│                         whether to capture videos of the agent performances (check out `videos` folder) (default: False)                           │
│ --log-frequency INT     the logging frequency of the algorithm (default: 1)                                                                        │
│ --log-examples-interval INT                                                                                                                        │
│                         the logging frequency of the examples (default: 20)                                                                        │
│ --resume, --no-resume   if toggled, tries to resume training from the latest checkpoint (default: False)                                           │
│ --ckpt-interval INT     the saving interval of the model (default: 1)                                                                              │
│ --overwrite-ckpt, --no-overwrite-ckpt                                                                                                              │
│                         if toggled and resuming is on, it will start fresh in resume mode, otherwise ignored (default: False)                      │
│ --env-id STR            the id of the environment (default: Uganda)                                                                                │
│ --num-envs INT          the number of parallel game environments (default: 4)                                                                      │
│ --agent {base_agent,llm_rules_agent,no_thoughts_agent}                                                                                             │
│                         the agent to use (default: llm_rules_agent)                                                                                │
│ --parallel-pipeline, --no-parallel-pipeline                                                                                                        │
│                         if toggled, the pipeline will be parallelized (default: True)                                                              │
│ --total-timesteps INT   total timesteps of the experiments (default: 1250)                                                                         │
│ --gamma FLOAT           the discount factor gamma (default: 0.95)                                                                                  │
│ --tau FLOAT             target smoothing coefficient (default: 1) (default: 1.0)                                                                   │
│ --batch-size INT        the batch size of sample from the reply memory (default: 32)                                                               │
│ --learning-starts INT   timestep to start learning (default: 64)                                                                                   │
│ --policy-lr FLOAT       the learning rate of the policy network optimizer (default: 0.0003)                                                        │
│ --q-lr FLOAT            the learning rate of the Q network network optimizer (default: 0.0003)                                                     │
│ --update-frequency FLOAT|INT                                                                                                                       │
│                         the frequency of training updates (default: 0.0625)                                                                        │
│ --warmup-updates INT    the number of warmup updates to the value function on the first iteration. (default: 64)                                   │
│ --target-network-frequency INT                                                                                                                     │
│                         the frequency of updates for the target networks (default: 64)                                                             │
│ --alpha FLOAT           Entropy regularization coefficient. (default: 0.01)                                                                        │
│ --autotune, --no-autotune                                                                                                                          │
│                         automatic tuning of the entropy coefficient (default: False)                                                               │
│ --target-entropy-scale FLOAT                                                                                                                       │
│                         coefficient for scaling the autotune entropy target (default: 0.89)                                                        │
│ --num-eval-steps INT    the number of steps to run in each eval environment per policy rollout (default: 64)                                       │
│ --eval-interval INT     the evaluation interval (default: 1)                                                                                       │
│ --eval-deterministic, --no-eval-deterministic                                                                                                      │
│                         if toggled, the evaluation will be deterministic (default: True)                                                           │
│ --rolling-rewards-window INT                                                                                                                       │
│                         the rolling rewards window (default: 64)                                                                                   │
│ --num-rules INT         The number of rules for rule-based LLM-only agent (default: 10)                                                            │
│ --llm                                                                                                                                              │
│ {google/gemma-2b-it,meta-llama/Llama-3.2-3B-Instruct-Turbo,meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo,meta-llama/Llama-3.3-70B-Instruct-Turbo,m… │
│                         the language model to use (default: gpt-4o-mini-huit)                                                                      │
│ --embedder-lm STR       the language model to use for embeddings (default: togethercomputer/m2-bert-80M-8k-retrieval)                              │
│ --embed-dim INT         the dimension of the embeddings (default: 768)                                                                             │
│ --hidden-dim INT        the hidden dimension of the networks (default: 16)                                                                         │
│ --buffer-collection-steps INT                                                                                                                      │
│                         the number of steps to collect data to the buffer (default: 64)                                                            │
│ --load-buffer, --no-load-buffer                                                                                                                    │
│                         if toggled, the agent will load the buffer from the pickle file if it exists (default: True)                               │
│ --buffer-size INT       the replay memory buffer size (default: 1024)                                                                              │
│ --compile-torch, --no-compile-torch                                                                                                                │
│                         Torch compile (default: False)                                                                                             
```




