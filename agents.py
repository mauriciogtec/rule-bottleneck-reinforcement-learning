from collections import defaultdict
import json
import re
from itertools import combinations
from typing import Callable, Dict, Sequence, List, Optional, Tuple

import torch
import torch.nn as nn
from gymnasium.core import ActType
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

import layers


def default_action_parser(s: str) -> ActType:
    out = [int(i) for i in re.findall(r"\d+", s)]
    return out[0] if len(out) == 1 else out


def generate_rule_combinations(
    rules: List[Dict], max_combs: Optional[int] = None
) -> List[Dict]:
    all_combinations = []
    for r in range(max_combs or len(rules)):
        for combs in combinations(rules, r + 1):
            all_combinations.append("\n".join(combs))

    return all_combinations


def parse_rules(x: str) -> List[str]:
    # 1. Remove the following patters
    patterns = ["```yaml", "```yml", "```"]
    x = re.sub("|".join(patterns), "", x)

    # 2. Remove trailing white space, and collapse new lines
    x = x.strip()
    x = re.sub(r"\n+", "\n", x)

    # 3. Break in lines
    x = x.split("\n")

    # 4. Remove empty lines
    x = [line for line in x if line.strip() != ""]

    return x


class BaseAgent:
    """base class for the CoT agent with explanation"""

    def __init__(
        self,
        task_text: str,
        action_space_text: str,
        llm: BaseChatModel,
        action_parser: Callable[[str], ActType] = default_action_parser,
    ):
        self.task_text = task_text
        self.action_space_text = action_space_text
        self.llm = llm
        self.action_parser = action_parser

    def __call__(
        self,
        state_text: str | list[str],
        state_vector: Optional[Sequence[float] | List[Sequence[float]]] = None,
        post_action: bool = True,
        **kwargs,
    ):
        # Initialize outputs and messages that will be updated by each pipeline step
        #    all intermediate outputs and finall results will be stored in the outputs
        #    all messages exchanged between the user and the llm assistant are stored in the messages

        # Call in a vectorized way when state_text is a list
        if isinstance(state_text, str):
            return self.pipeline(
                state_text, state_vector=state_vector, post_action=post_action, **kwargs
            )
        else:
            # call for each
            all_actions = []
            all_messages = []
            all_outputs = defaultdict(list)
            for i in range(len(state_text)):
                action, outputs, messages = self.pipeline(
                    state_text[i],
                    state_vector=state_vector[i] if state_vector is not None else None,
                    post_action=post_action,
                    **kwargs,
                )
                all_actions.append(action)
                all_messages.append(messages)
                for k, v in outputs.items():
                    all_outputs[k].append(v)

            return all_actions, dict(all_outputs), all_messages

    def pipeline(
        self,
        state_text: str,
        state_vector: Optional[Sequence[float]] = None,
        post_action: bool = True,
        **kwargs,
    ) -> Tuple[ActType, Dict, List[Dict]]:
        initial_prompt = self.system_prompt_with_state(state_text)
        outputs = {
            "state_text": state_text,
            "state_vector": state_vector,
            "initial_prompt": initial_prompt,
            **kwargs,
        }
        messages = [{"role": "system", "content": initial_prompt}]

        # Pre-action step (e.g., stores thoughts in the outputs)
        self.pre_action(outputs, messages)

        # Get action (e.g., asks the user to choose an action and parses the response)
        action = self.get_action(outputs, messages)

        # Post-action step (e.g., stores explanation in the outputs)
        if post_action:
            self.post_action(outputs, messages)

        return action, outputs, messages

    def system_prompt_with_state(self, state_text: str) -> str:
        return (
            f"### Task\n\n {self.task_text}"
            f"\n\n### Possible actions\n\n {self.action_space_text}"
            f"\n\n### Current state of the environment\n\n {state_text}\n"
        )

    def pre_action(self, outputs: Dict, messages: List[Dict]):
        """Initializes outputs and messages"""
        self.gen_thoughts(outputs, messages)

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        # get actions
        action_prompt = (
            "Now, choose the optimal action given the current state of the environment and the set of priorization rules. "
            "Do not provide additional information or context for your answer, only the action as follows. "
            f"\n\n### Possible actions:\n\n {self.action_space_text}"
            "\n\n### Your response:"
        )
        messages += [{"role": "user", "content": action_prompt}]

        outputs["action_str"] = self.llm.invoke(messages, max_tokens=10).content
        messages.append({"role": "assistant", "content": outputs["action_str"]})

        outputs["action"] = self.action_parser(outputs["action_str"])
        return outputs["action"]

    def post_action(self, outputs: Dict, messages: List[Dict]) -> None:
        """Finalizes outputs and messages"""
        self.gen_explanation(outputs, messages)

    def gen_thoughts(self, outputs: Dict, messages: List[Dict]) -> None:
        """Generate thoughts and update message list"""

        thought_prompt = (
            "First, reason about what elements should be considered when choosing the optimal action"
            and " in the given task of the decision making agent."
            " Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks"
            " of each action in the current state. Conclude the paragraph with a reflection of how they inform the design"
            " of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
        )

        messages.append({"role": "user", "content": thought_prompt})
        outputs["thoughts"] = self.llm.invoke(
            messages, temperature=0.5, max_tokens=200
        ).content
        messages.append({"role": "assistant", "content": outputs["thoughts"]})

    def gen_explanation(self, outputs: Dict, messages: List[Dict]) -> str:
        """Generate explanation and update message list"""
        explanation_prompt = (
            "Explain why you chose the optimal action given the current state of the environment and the set of priorization rules. "
            "Your response should be a short paragraph with 1-3 sentences that explain the reasoning behind your choice."
        )

        messages.append({"role": "user", "content": explanation_prompt})
        outputs["explanation"] = self.llm.invoke(
            messages, temperature=0.5, max_tokens=200
        ).content
        messages.append({"role": "assistant", "content": outputs["explanation"]})


class LLMRulesAgent(BaseAgent):
    """The rule-based agent generates a set of rules in addition to thoughts"""

    def __init__(
        self,
        task_text: str,
        action_space_text: str,
        llm: BaseChatModel,
        num_rules: int = 5,
        example_rules: Optional[str] = None,
        max_parse_attempts: int = 3,
        verbose: bool = False,
        action_parser: Callable[[str], ActType] = default_action_parser,
    ):
        super().__init__(
            task_text=task_text,
            action_space_text=action_space_text,
            llm=llm,
            action_parser=action_parser,
        )
        self.num_rules = num_rules
        self.example_rules = example_rules
        self.max_parse_attempts = max_parse_attempts
        self.verbose = verbose

    def pre_action(self, outputs: Dict, messages: List[Dict]):
        self.gen_thoughts(outputs, messages)
        self.gen_rules(outputs, messages)

    def gen_rules(self, outputs: Dict, messages: List[Dict]):

        rules_prompt = (
            f"Now, suggest {self.num_rules} rules that could be useful to make an optinal decision in the current state. "
            " For each rule, provide the explanation of why it is important to consider it at the given state."
            " Your response consist solely of a machine-readable YAML list."
            " Each rule should be exactly one line and start with the character `-`."
            " The rules should be in natural language. Follow the following tempalte:'Because of [short explanation], prioritize [something] [if/when]. [Explanation]."
            " The 'Explanation' should elaborate on the expected outcome of following the rule and its connection with "
            " the task and the agent's goals."
            " Your answer should start with the character ```- "
        )

        # send second call using the OpenAI API
        messages.append({"role": "user", "content": rules_prompt})
        response = self.llm.invoke(messages, max_tokens=200).content
        outputs["rules"] = parse_rules(response)
        messages.append({"role": "assistant", "content": outputs["rules"]})


class RulesSelectorActorCritic(BaseAgent):
    """The rule-based agent generates a set of rules based on the environment state."""

    def __init__(
        self,
        actor_critic: layers.AttentionActorCritic,
        task_text: str,
        action_space_text: str,
        llm: BaseChatModel,
        embededder: Embeddings,
        max_rule_combinations: int = 3,
        num_rules: int = 5,
        example_rules: Optional[str] = None,
        max_parse_attempts: int = 3,
        verbose: bool = False,
        action_parser: Callable[[str], ActType] = default_action_parser,
    ):
        super().__init__(
            task_text=task_text,
            action_space_text=action_space_text,
            llm=llm,
            action_parser=action_parser,
        )
        assert isinstance(actor_critic, layers.AttentionActorCritic)

        self.actor_critic = actor_critic
        self.max_rule_combinations = max_rule_combinations
        self.embedder = embededder
        self.num_rules = num_rules
        self.example_rules = example_rules
        self.max_parse_attempts = max_parse_attempts
        self.verbose = verbose

    def pre_action(self, outputs: Dict, messages: List[Dict]):
        self.gen_thoughts(outputs, messages)
        self.gen_rules(outputs, messages)

    def gen_rules(self, outputs: dict, messages: list[dict]) -> List[str]:
        """Wrapper for generating rules that includes combinations of and embeddings for rules.
        First, generates combinations of rules, next it filters using an RL selector
        """
        # get the state, throw error if state is not present
        assert (
            outputs["state_vector"] is not None
        ), "passing state_vector when calling the agent is required."
        state_vector = outputs["state_vector"]
        device = state_vector.device

        # get all possible rules with combinations
        rules_prompt = (
            f"Now, suggest {self.num_rules} rules that could be useful to solve the task. "
            " For each rule, provide the explanation of why it is important to consider it at the given state."
            " Your response consist solely of a machine-readable YAML list."
            " Your response should be a list of rules. Each rule should be exactly one line and start with the character `-`."
            " The rules should be in natural language. While there is no strict format, it is recommended "
            " that they follow the following tempalte:'Because of [short explanation], prioritize [something] [if/when]. [Explanation]."
            " The 'Explanation' should elaborate on the expected outcome of following the rule and its connection with "
            " the task and the agent's goals."
            " Your answer should start with the character ```-  and end with ```"
        )

        # send second call using the OpenAI API
        messages.append({"role": "user", "content": rules_prompt})
        response = self.llm.invoke(messages, max_tokens=512).content

        rules = parse_rules(response)
        outputs["rules"] = rules = generate_rule_combinations(
            rules, max_combs=self.max_rule_combinations
        )

        # get the rules and state embeddings
        # rules_emb = self._generate_embeddings_for_rules(rules)
        rules_emb = self.embedder.embed_documents(rules)
        rules_emb = torch.tensor(rules_emb, dtype=torch.float32).to(device)
        outputs["rules_emb"] = rules_emb

        # query (1, emb_dim,) keys (num_rules, emb_dim)
        query, keys = state_vector.unsqueeze(0), rules_emb

        # logits (1, num_rules) -> (num_rules, )
        logits, value = self.actor_critic(query, keys)

        dist = torch.distributions.Categorical(logits=logits)
        sel_idx = dist.sample().squeeze()
        entropy = dist.entropy().squeeze()

        # get the selected rule
        sel_rule = rules[sel_idx]

        outputs["logits"] = logits
        outputs["sel_logprob"] = dist.log_prob(sel_idx).squeeze()
        outputs["sel_idx"] = sel_idx
        outputs["sel_rule"] = sel_rule
        outputs["entropy"] = entropy
        outputs["value"] = value.squeeze()

    def gen_rule_scores(self, outputs: Dict, messages: List[Dict]) -> Dict:
        """Generate rule scores based on the current state and the set of rules."""
        rule = outputs["sel_rule"]

        # dummy implementation
        sel_reward = torch.tensor(-len(rule) / 1000, dtype=torch.float32)
        device = outputs["state_vector"].device
        outputs["sel_reward"] = sel_reward.to(device)

    def post_action(self, outputs, messages):
        self.gen_explanation(outputs, messages)
        self.gen_rule_scores(outputs, messages)

    def get_action_and_value(
        self,
        state_vector: torch.Tensor,
        rules_emb: torch.Tensor,
        rules_padding_mask: Optional[torch.Tensor] = None,
        sel_idxs: Optional[torch.Tensor] = None,
    ):
        logits, values = self.actor_critic(
            state_vector.unsqueeze(1), rules_emb, key_padding_mask=rules_padding_mask
        )
        dist = torch.distributions.Categorical(logits=logits)
        if sel_idxs is None:
            sel_idxs = dist.sample()
        entropy = dist.entropy()
        log_prob = dist.log_prob(sel_idxs)

        return sel_idxs, log_prob, entropy, values
