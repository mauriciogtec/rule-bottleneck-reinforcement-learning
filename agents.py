from collections import defaultdict
import json
import re
from itertools import combinations
from typing import Callable, Dict, Sequence, List, Optional, Tuple

import numpy as np
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
    ):
        self.task_text = task_text
        self.action_space_text = action_space_text
        self.llm = llm

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

        # outputs is dictionary that collects all intermediate outputs
        # by all steps of the pipeline
        outputs = {
            "state_text": state_text,
            "state_vector": state_vector,
            "initial_prompt": initial_prompt,
            **kwargs,
        }

        # messages collects the conversation between the user and the LLM
        # in the format required by the openai API
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
            f"\n\n### Possible actions\n\n{self.action_space_text}"
            f"\n\n### Current state of the decision problem\n\n{state_text}"
        )

    def pre_action(self, outputs: Dict, messages: List[Dict]):
        """Initializes outputs and messages"""
        self.gen_thoughts(outputs, messages)

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        # get actions
        action_prompt = (
            "Now, choose the optimal action given the current state of the decision problem. "
            "Do not provide additional information or context for your answer, only the action as follows. "
            f"\n\n### Possible actions:\n\n{self.action_space_text}"
            # "\n\n### Your response:"
        )
        messages.append({"role": "user", "content": action_prompt})

        outputs["action"] = self.llm.invoke(messages, max_tokens=10).content
        messages.append({"role": "assistant", "content": outputs["action"]})

        # outputs["action"] = self.action_parser(outputs["action_str"])
        return outputs["action"]

    def post_action(self, outputs: Dict, messages: List[Dict]):
        """Finalizes outputs and messages"""
        self.gen_explanation(outputs, messages)
        self.gen_explanation_no_thoughts(outputs, messages)

    def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
        """Generate thoughts and update message list"""

        thought_prompt = (
            "First, reason about what elements should be considered when choosing the optimal action"
            " in the given task of the decision making agent."
            " Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks"
            " of each action in the current state. Conclude the paragraph with a reflection of how they inform the design"
            " of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
        )

        messages.append({"role": "user", "content": thought_prompt})
        outputs["thoughts"] = self.llm.invoke(
            messages, temperature=0.5, max_tokens=200
        ).content
        messages.append({"role": "assistant", "content": outputs["thoughts"]})

    def gen_explanation(self, outputs: Dict, messages: List[Dict]):
        """Generate explanation and update message list"""
        explanation_prompt = (
            "Explain why you chose the optimal action based on the previous considerations. "
            "Your response should be a short paragraph with 1-3 sentences that explain the reasoning behind your choice."
        )

        messages.append({"role": "user", "content": explanation_prompt})
        outputs["explanation"] = self.llm.invoke(
            messages, temperature=0.5, max_tokens=200
        ).content
        messages.append({"role": "assistant", "content": outputs["explanation"]})

    def gen_explanation_no_thoughts(self, outputs: Dict, messages: List[Dict]) -> str:
        """Generate explanation and update message list"""
        temp_system_prompt = f"{self.system_prompt_with_state(outputs['state_text'])}"
        explanation_prompt = (
            "Explain why you chose these action."
            "Your response should be a short paragraph with 1-3 sentences that explain the reasoning behind your choice."
        )

        temp_messages = [
            {"role": "system", "content": temp_system_prompt},
            {"role": "user", "content": explanation_prompt},
        ]

        response = self.llm.invoke(temp_messages, max_tokens=200).content
        outputs["explanation_no_thoughts"] = response


class NoThoughtsAgent(BaseAgent):
    def pre_action(self, outputs: Dict, messages: List[Dict]):
        pass


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
    ):
        super().__init__(
            task_text=task_text,
            action_space_text=action_space_text,
            llm=llm,
        )
        self.num_rules = num_rules
        self.example_rules = example_rules
        self.max_parse_attempts = max_parse_attempts
        self.verbose = verbose

    def pre_action(self, outputs: Dict, messages: List[Dict]):
        self.gen_thoughts(outputs, messages)
        self.gen_rules(outputs, messages)

    def post_action(self, outputs: Dict, messages: List[Dict]):
        self.gen_explanation(outputs, messages)
        self.gen_explanation_rule_only(outputs, messages)
        self.gen_rule_scores(outputs, messages)

    def gen_explanation_rule_only(self, outputs, messages):
        """Generate rule scores based on the current state and the set of rules."""

        rules = "\n".join(outputs["rules"])
        temp_system_prompt = (
            f"{self.system_prompt_with_state(outputs['state_text'])}"
            "To solve the task above, an algorithm has suggested the following priorization rules:\n\n"
            f"### Selected rule\n\n{rules}\n\n"
        )
        explanation_prompt = (
            "Explain why you chose these action. Does the rule explain it?"
            "Your response should be a short paragraph with 1-3 sentences that explain the reasoning behind your choice."
        )

        temp_messages = [
            {"role": "system", "content": temp_system_prompt},
            {"role": "user", "content": explanation_prompt},
        ]

        response = self.llm.invoke(temp_messages, max_tokens=200).content
        outputs["explanation_rule_only"] = response

    def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
        thought_prompt = (
            "First, reason about what elements should be considered when choosing the optimal action"
            " in the given task of the decision making agent."
            " Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks"
            " of each action in the current state. Conclude the paragraph with a reflection of how they inform the design"
            " of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
        )
        messages.append({"role": "user", "content": thought_prompt})
        outputs["thoughts"] = self.llm.invoke(
            messages, temperature=0.5, max_tokens=200
        ).content
        messages.append({"role": "assistant", "content": outputs["thoughts"]})

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        # get actions
        action_prompt = (
            "Now, choose the optimal action given the current state of the decision problem and the decision rules. "
            "Do not provide additional information or context for your answer, only the action as follows. "
            f"\n\n### Possible actions:\n\n{self.action_space_text}"
        )
        messages.append({"role": "user", "content": action_prompt})

        outputs["action"] = self.llm.invoke(messages, max_tokens=10).content
        messages.append({"role": "assistant", "content": outputs["action"]})

        return outputs["action"]

    def gen_rules(self, outputs: Dict, messages: List[Dict]):
        rules_prompt = (
            f"Now, suggest {self.num_rules} rules that could be useful to make an optimal decision in the current state. "
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
        rules_str = "\n".join(outputs["rules"])
        messages.append({"role": "assistant", "content": rules_str})

    def gen_rule_scores(self, outputs: Dict, messages: List[Dict]):
        """Generate rule scores based on the current state and the set of rules."""
        # Get only the task prompt with state and prompt the LLM to check whether the
        # selected rule is enough to understand
        # 1) What will the agent to next (action prediction)?
        # 2) Is the rule relevant to the current state?
        # 3) Is the justification of the rule clear and relates to the task?

        rules = "\n".join(["- " + x for x in outputs["rules"]])
        temp_system_prompt = (
            f"{self.system_prompt_with_state(outputs['state_text'])}"
            "To solve the task above, an algorithm has suggested the following priorization rules:\n\n"
            f"### Selected rules\n\n{rules}\n\n"
            "You will be given a series of questions you need to answer with a simple 'yes' or 'no'"
            " without any additional information or justification. Your response should be a single word."
        )
        q1 = "1. Are the rules sufficient to understand what action should you (the agent) do next?"
        q2 = "2. Are the rules applicable to the current state of the decision problem?"
        q3 = "3. Are the rules appropriately justified, without fallacies or hallucination?"
        q4 = f"4. The agent chose action {outputs['action']} based on the problem state. Are the selected rules sufficient to explain the decision?"
        q5 = "5. Below is the explanation by the agent for the selected action. Rate it in a scale from 1 to 10."

        coda = "\nAnswer the following questions with a simple 'yes' or 'no'.\n\n### Your response:"
        coda2 = f"\n\n{outputs['explanation']}\n\nRespond with a single number from 1 to 10 without any additional information.\n\n### Your response:"

        # Answer q1
        temp_messages = [
            {"role": "system", "content": temp_system_prompt},
            {"role": "user", "content": q1 + coda},
        ]
        r1_ = self.llm.invoke(temp_messages, max_tokens=2).content
        r1 = float("yes" in r1_.lower())
        temp_messages.append({"role": "assistant", "content": r1_})

        # Answer q2
        temp_messages.append({"role": "user", "content": q2 + coda})
        r2_ = self.llm.invoke(temp_messages, max_tokens=2).content
        r2 = float("yes" in r2_.lower())
        temp_messages.append({"role": "assistant", "content": r2_})

        # Answer q3
        temp_messages.append({"role": "user", "content": q3 + coda})
        r3_ = self.llm.invoke(temp_messages, max_tokens=2).content
        r3 = float("yes" in r3_.lower())
        temp_messages.append({"role": "assistant", "content": r3_})

        # Answer q4
        q4_with_action = q4 + f"Selected action: {outputs['action']}\n\n"
        q4_with_action += (
            f"Post-hoc explanation: {outputs['explanation_rule_only']}\n\n"
        )
        temp_messages.append({"role": "user", "content": q4_with_action + coda})
        r4_ = self.llm.invoke(temp_messages, max_tokens=2).content
        r4 = float("yes" in r4_.lower())
        temp_messages.append({"role": "assistant", "content": r4_})

        # Answer q5
        temp_messages.append({"role": "user", "content": q5 + coda2})
        r5_ = self.llm.invoke(temp_messages, max_tokens=2).content
        try:
            r5 = float(re.findall(r"\d+", r5_)[0]) / 10  # get the first number
        except:
            # random
            r5 = np.random.rand()
        temp_messages.append({"role": "assistant", "content": r5_})

        # Calculate the reward
        outputs["sel_reward"] = float(np.mean([r1, r2, r3, r4]))
        outputs["sel_reward_scores"] = [r1, r2, r3, r4, r5]
        outputs["sel_reward_scores_raw"] = {q1: r1_, q2: r2_, q3: r3_, q4: r4_, q5: r5_}


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
    ):
        super().__init__(
            task_text=task_text,
            action_space_text=action_space_text,
            llm=llm,
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
        tmp_messages = messages + [{"role": "user", "content": rules_prompt}]
        response = self.llm.invoke(tmp_messages, max_tokens=512).content

        rules = parse_rules(response)
        outputs["rules"] = rules = generate_rule_combinations(
            rules, max_combs=self.max_rule_combinations
        )
        rules_str = "\n".join(rules)

        # dont' add all rules, confuses the LLM and increases the cost
        # messages.append({"role": "assistant", "content": rules_str})

        # get the rules and state embeddings
        # rules_emb = self._generate_embeddings_for_rules(rules)
        rules_emb = self.embedder.embed_documents(rules)
        rules_emb = torch.tensor(rules_emb, dtype=torch.float32).to(device)
        outputs["rules_emb"] = rules_emb

        # get the rule scores
        query, keys = state_vector.unsqueeze(0), rules_emb
        logits, value = self.actor_critic(query, keys)
        logits = logits.squeeze(0)
        value = value.squeeze()

        dist = torch.distributions.Categorical(logits=logits)
        sel_idx = dist.sample()
        entropy = dist.entropy()

        # get the selected rule
        sel_rule = rules[sel_idx]

        outputs["logits"] = logits
        outputs["sel_logprob"] = dist.log_prob(sel_idx)
        outputs["sel_idx"] = sel_idx
        outputs["sel_rule"] = sel_rule
        outputs["entropy"] = entropy
        outputs["value"] = value

    def gen_rule_scores(self, outputs: Dict, messages: List[Dict]):
        """Generate rule scores based on the current state and the set of rules."""
        # Get only the task prompt with state and prompt the LLM to check whether the
        # selected rule is enough to understand
        # 1) What will the agent to next (action prediction)?
        # 2) Is the rule relevant to the current state?
        # 3) Is the justification of the rule clear and relates to the task?

        sel_rule = outputs["sel_rule"]
        temp_system_prompt = (
            f"{self.system_prompt_with_state(outputs['state_text'])}"
            "To solve the task above, an algorithm has suggested the following priorization rule:\n\n"
            f"### Selected rule\n\n{sel_rule}\n\n"
            "You will be given a series of questions you need to answer with a simple 'yes' or 'no'"
            " without any additional information or justification. Your response should be a single word."
        )
        q1 = "1. Is the rule sufficient to understand what action should you (the agent) do next?"
        q2 = "2. Is the rule applicable to the current state of the decision problem?"
        q3 = "3. Is the rule appropriately justified, without fallacies or hallucination?"
        q4 = f"4. The agent chose action {outputs['action']} based on the problem state. Is the selected rule sufficient to explain it?"
        q5 = "5. Below is the explanation by the agent for the selected action. Rate it in a scale from 1 to 10."

        coda = "\nAnswer the following questions with a simple 'yes' or 'no'.\n\n### Your response:"
        coda2 = f"\n\n{outputs['explanation']}\n\nRespond with a single number from 1 to 10 without any additional information.\n\n### Your response:"

        # Answer q1
        temp_messages = [
            {"role": "system", "content": temp_system_prompt},
            {"role": "user", "content": q1 + coda},
        ]
        r1_ = self.llm.invoke(temp_messages, max_tokens=2).content
        r1 = float("yes" in r1_.lower())
        temp_messages.append({"role": "assistant", "content": r1_})

        # Answer q2
        temp_messages.append({"role": "user", "content": q2 + coda})
        r2_ = self.llm.invoke(temp_messages, max_tokens=2).content
        r2 = float("yes" in r2_.lower())
        temp_messages.append({"role": "assistant", "content": r2_})

        # Answer q3
        temp_messages.append({"role": "user", "content": q3 + coda})
        r3_ = self.llm.invoke(temp_messages, max_tokens=2).content
        r3 = float("yes" in r3_.lower())
        temp_messages.append({"role": "assistant", "content": r3_})

        # Answer q4
        q4_with_action = q4 + f"Selected action: {outputs['action']}\n\n"
        q4_with_action += (
            f"Post-hoc explanation: {outputs['explanation_rule_only']}\n\n"
        )
        temp_messages.append({"role": "user", "content": q4_with_action + coda})
        r4_ = self.llm.invoke(temp_messages, max_tokens=2).content
        r4 = float("yes" in r4_.lower())
        temp_messages.append({"role": "assistant", "content": r4_})

        # Answer q5
        temp_messages.append({"role": "user", "content": q5 + coda2})
        r5_ = self.llm.invoke(temp_messages, max_tokens=2).content
        try:
            r5 = float(re.findall(r"\d+", r5_)[0]) / 10  # get the first number
        except:
            # random
            r5 = np.random.rand()
        temp_messages.append({"role": "assistant", "content": r5_})

        # Calculate the reward
        sel_reward = np.mean([r1, r2, r3, r4])
        device = outputs["state_vector"].device
        sel_reward = torch.tensor(sel_reward, dtype=torch.float32).to(device)
        outputs["sel_reward"] = sel_reward

        outputs["sel_reward_scores"] = [r1, r2, r3, r4, r5]
        outputs["sel_reward_scores_raw"] = {q1: r1_, q2: r2_, q3: r3_, q4: r4_, q5: r5_}

    def post_action(self, outputs, messages):
        self.gen_explanation(outputs, messages)
        self.gen_explanation_rule_only(outputs, messages)
        self.gen_rule_scores(outputs, messages)

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        # get actions
        action_prompt = (
            f"### Selected priorization rules\n\nBelow are the rules that could be useful to make an optimal decision in the current state:\n\n"
            f"{outputs['sel_rule']}\n\n"
            "### The decision\n\n"
            "Now, choose the optimal action given the current state of the decision problem and the chosen priorization rules."
            " Your decision should be made considering the thoughts and selected priorization rules."
            " Your answer should be one of the valid actions described below "
            " without additional information or justification. Your response start withand only consist of the action.\n\n"
            f"\n\n{self.action_space_text}\n\n"
            "### Your response:"  # This somehow helps with the instruction following
        )
        messages.append({"role": "user", "content": action_prompt})

        outputs["action"] = self.llm.invoke(messages, max_tokens=10).content
        messages.append({"role": "assistant", "content": outputs["action"]})
        return outputs["action"]

    def get_action_and_value_from_embeddings(
        self,
        state_vector: torch.Tensor,
        rules_emb: torch.Tensor,
        rules_padding_mask: Optional[torch.Tensor] = None,
        sel_idxs: Optional[torch.Tensor] = None,
    ):
        logits, values = self.actor_critic(
            state_vector.unsqueeze(1), rules_emb, key_padding_mask=rules_padding_mask
        )
        logits = logits.squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)
        if sel_idxs is None:
            sel_idxs = dist.sample()
        entropy = dist.entropy()
        log_prob = dist.log_prob(sel_idxs)

        return sel_idxs, log_prob, entropy, values

    def gen_explanation_rule_only(self, outputs, messages):
        """Generate rule scores based on the current state and the set of rules."""

        sel_rule = outputs["sel_rule"]
        temp_system_prompt = (
            f"{self.system_prompt_with_state(outputs['state_text'])}"
            "To solve the task above, an algorithm has suggested the following priorization rule:\n\n"
            f"### Selected rule\n\n{sel_rule}\n\n"
        )
        explanation_prompt = (
            "### Question\n\n"
            "Explain why you chose these action. Does the rule explain it?"
            "Your response should be a short paragraph with 1-3 sentences that explain the reasoning behind your choice."
        )

        temp_messages = [
            {"role": "system", "content": temp_system_prompt},
            {"role": "user", "content": explanation_prompt},
        ]

        response = self.llm.invoke(temp_messages, max_tokens=200).content
        outputs["explanation_rule_only"] = response
