from functools import partial
import os
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from gymnasium.core import ActType
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

import layers
from llm_apis import HFMetaWrapper, invoke_with_retries

PureLanguageAgents = Literal[
    "base_agent", "llm_rules_agent", "no_thoughts_agent", "llm_rules_no_thoughts"
]


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
    patterns = ["```yaml", "```yml", "```", "```json", "json", ]
    x = re.sub("|".join(patterns), "", x)

    # 2. Remove trailing white space, and collapse new lines
    x = x.strip()
    x = re.sub(r"\n+", "\n", x)

    # 3. Break in lines
    x = x.split("\n")

    # Get all the content wrapped between braces {}
    # x = re.findall(r"\{([^}]+)\}", x)

    # 4. Remove empty lines
    x = [line for line in x if line.strip() != ""]

    # Make sure there are no line breaks inside brances {}
    # Any match of the form {xxxx\nxxxx} is replaced by {xxxx xxxx}
    x = [
        re.sub(r"\{([^}]+)\}", lambda m: m.group(0).replace("\n", " "), line)
        for line in x
    ]

    # 5. Add the braces back
    # x = [f"{{{line}}}" for line in x]
    return x


class BaseAgent:
    """base class for the CoT agent with explanation"""

    def __init__(
        self,
        task_text: str,
        action_space_text: str,
        llm: BaseChatModel,
        use_thoughts: bool = True,
    ):
        self.task_text = task_text
        self.action_space_text = action_space_text
        self.llm = llm
        self.use_thoughts = use_thoughts

    def pipeline(
        self,
        state_text: str,
        state_vector: Optional[Sequence[float]] = None,
        pre_action_only: bool = False,
        include_post_action: bool = True,
        pre_action_outputs: Optional[Dict] = None,
        pre_action_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Tuple[ActType, Dict, List[Dict]]:
        """Runs the pipeline for the agent"""
        initial_prompt = self.system_prompt_with_state(state_text)

        # check that either both prev_outputs and prev_messages are None or both are not None
        if pre_action_outputs is not None:
            outputs = pre_action_outputs
            messages = pre_action_messages

        else:
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

            if pre_action_only:
                return outputs, messages

        # Get action (e.g., asks the user to choose an action and parses the response)
        self.get_action(outputs, messages)

        # Post-action step (e.g., stores explanation in the outputs)
        if include_post_action:
            self.post_action(outputs, messages)

        return outputs, messages

    def parallel_pipeline(
        self,
        state_text: List[str],
        state_vector: Optional[Sequence[Sequence[float]]] = None,
        post_action: bool = True,
        num_procs: Optional[int] = None,
        pre_action_only: bool = False,
        include_post_action: bool = True,
        pre_action_outputs: Optional[List[Dict]] = None,
        pre_action_messages: Optional[List[List[Dict]]] = None,
    ):
        if state_vector is None:
            state_vector = [None] * len(state_text)

        if pre_action_outputs is None:
            pre_action_outputs = [None] * len(state_text)

        if pre_action_messages is None:
            pre_action_messages = [None] * len(state_text)

        # Get the action and value in parallel
        def call_pipeline(i):

            with torch.no_grad():
                return self.pipeline(
                    state_text=state_text[i],
                    state_vector=state_vector[i],
                    post_action=post_action,
                    pre_action_only=pre_action_only,
                    include_post_action=include_post_action,
                    pre_action_outputs=pre_action_outputs[i],
                    pre_action_messages=pre_action_messages[i],
                )

        num_envs = len(state_text)
        if num_procs is None:
            num_procs = os.cpu_count() - 1
        with ThreadPoolExecutor(max_workers=num_procs) as executor:
            results = list(executor.map(call_pipeline, range(num_envs)))
            outputs, messages = zip(*results)
            # outputs = {key: [output[key] for output in outputs] for key in outputs[0]}
            messages = list(messages)
            outputs = list(outputs)

        return outputs, messages

    def system_prompt_with_state(self, state_text: str) -> str:
        return (
            f"### Task\n\n {self.task_text}"
            f"\n\n### Possible actions\n\n{self.action_space_text}"
            f"\n\n### Current state of the decision problem\n\n{state_text}"
        )

    def pre_action(self, outputs: Dict, messages: List[Dict]):
        """Initializes outputs and messages"""
        if self.use_thoughts:
            self.gen_thoughts(outputs, messages)

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        # get actions
        action_prompt = (
            "Now, choose the optimal action given the current problem state. "
            "Do not provide additional information or context for your answer, only the action as follows. "
            f"\n\n### Possible actions:\n\n{self.action_space_text}"
            # "\n\n### Your response:"
        )
        messages.append({"role": "user", "content": action_prompt})

        outputs["action"] = invoke_with_retries(
            self.llm, messages, max_tokens=10
        ).content
        messages.append({"role": "assistant", "content": outputs["action"]})

        # outputs["action"] = self.action_parser(outputs["action_str"])
        return outputs["action"]

    def post_action(self, outputs: Dict, messages: List[Dict]):
        """Finalizes outputs and messages"""
        self.gen_explanation(outputs, messages)

    def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
        prompt = (
            "First, reason about what elements should be considered when choosing the optimal action."
            " Your response should consist of a single short paragraph that reflects on the consequences, benefits, and drawbacks"
            " of each action in the current state."
        )
        messages.append({"role": "user", "content": prompt})
        outputs["thoughts"] = invoke_with_retries(
            self.llm, messages, temperature=0.5, max_tokens=256
        ).content
        messages.append({"role": "assistant", "content": outputs["thoughts"]})


    def gen_explanation(self, outputs: Dict, messages: List[Dict]):
        return _gen_explanation(outputs, messages, self.llm)


NoThoughtsAgent = partial(BaseAgent, use_thoughts=False)


def _gen_rule_scores(outputs, messages, llm, rules, system_prompt):
    """Generate rule scores based on the current state and the set of rules."""
    # Get only the task prompt with state and prompt the LLM to check whether the
    # selected rule is enough to understand
    # 1) What will the agent to next (action prediction)?
    # 2) Is the rule relevant to the current state?
    # 3) Is the justification of the rule clear and relates to the task?

    rules = "\n".join(rules)
    rule_scores_prompt = (
        "To make a decision in the current state, the following rule/rules was/were selected:\n\n"
        f"\n\n{rules}\n\n"
        "You will now be given a question you need to answer with a simple 'yes' or 'no'.\n\n"
    )
    q1 = "Is/are the rule/rules **alone** sufficient to understand the optimal action/decision that the system should take in current the problem state?"
    q2 = "Is the condition in the rule/rules actionable and complete in the current problem state (containing sufficient detail about the current problem state without unnecessary information)?"
    q3 = "Did the selected rule/rules sufficiently help to understand the previous decision without contradictions?"
    q4 = "Is the justification of the rule satisfactory without false logic or hallucinations?"

    coda = (
        "\nAnswer the following questions with a simple 'yes' or 'no' without additional"
        " information or justification. Your response should be a single word.\n\n"
    )

    # Answer q1
    temp_messages = messages.copy()
    msg = rule_scores_prompt + "### Question\n\n" + q1 + coda
    temp_messages.append({"role": "user", "content": msg})
    r1_ = invoke_with_retries(llm, temp_messages, max_tokens=2).content
    r1 = float("yes" in r1_.lower())
    temp_messages.append({"role": "assistant", "content": r1_})

    # Answer q2
    temp_messages = messages.copy()
    msg = rule_scores_prompt + "### Question\n\n" + q2 + coda
    temp_messages.append({"role": "user", "content": msg})
    r2_ = invoke_with_retries(llm, temp_messages, max_tokens=2).content
    r2 = float("yes" in r2_.lower())
    temp_messages.append({"role": "assistant", "content": r2_})

    # Answer q3
    temp_messages = messages.copy()
    msg = rule_scores_prompt + (
        f"The decision taken in the current problem state was: {outputs['action']}.\n\n"
        f"### Question\n\n{q3 + coda}"
    )
    temp_messages.append({"role": "user", "content": msg})
    r3_ = invoke_with_retries(llm, temp_messages, max_tokens=2).content
    r3 = float("yes" in r3_.lower())
    temp_messages.append({"role": "assistant", "content": r3_})

    # Answer q4
    temp_messages = messages.copy()
    msg = rule_scores_prompt + "### Question\n\n" + q4 + coda
    temp_messages.append({"role": "user", "content": msg})
    r4_ = invoke_with_retries(llm, temp_messages, max_tokens=2).content
    r4 = float("yes" in r4_.lower())
    temp_messages.append({"role": "assistant", "content": r4_})


    # # Answer q5
    # temp_messages.append({"role": "user", "content": q5 + coda2})
    # r5_ = invoke_with_retries(self.llm, temp_messages, max_tokens=2).content
    # try:
    #     r5 = float(re.findall(r"\d+", r5_)[0]) / 10  # get the first number
    # except:
    #     # random
    #     r5 = np.random.rand()
    # temp_messages.append({"role": "assistant", "content": r5_})

    # Calculate the reward"]
    outputs["sel_reward"] = float(np.mean([r1, r2, r3, r4]))
    outputs["sel_reward_scores"] = [r1, r2, r3, r4]
    outputs["sel_reward_scores_raw"] = {q1: r1_, q2: r2_, q3: r3_, q4: r4_}


def _gen_thoughts_for_rule_agents(outputs, messages, llm, save_prompts: bool = True):
    # prompt = (
    #     "First, reason about what elements should be considered when choosing the optimal action"
    #     " in the given task of the decision making agent."
    #     " Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks"
    #     " of each action in the current state. Conclude the paragraph with a reflection of how they inform the design"
    #     " of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
    # )
    prompt = (
        "First, reason about what elements should be considered when choosing the optimal action."
        " Your response should consist of a single short paragraph that reflects on the consequences, benefits, and drawbacks"
        " of each action in the current state."
    )
    tmp_messages = messages.copy()
    tmp_messages.append({"role": "user", "content": prompt})
    response = invoke_with_retries(llm, tmp_messages, temperature=0.5, max_tokens=256).content
    
    if save_prompts:
        outputs["thoughts"] = response
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": outputs["thoughts"]})

    return response
    


def _gen_explanation(outputs, messages, llm):
    """Generate explanation and update message list"""
    explanation_prompt = (
        f"You chose action {outputs['action']} in the current problem state. "
        "Explain why you chose the optimal action based on the conversation. "
        "Your response should be a short paragraph with 1-3 sentences that explain the reasoning behind your choice."
    )

    messages.append({"role": "user", "content": explanation_prompt})
    outputs["explanation"] = invoke_with_retries(
        llm, messages, temperature=0.5, max_tokens=200
    ).content
    messages.append({"role": "assistant", "content": outputs["explanation"]})


def _gen_rules(
    outputs, messages, llm, num_rules=5, example_rules=None, save_prompts: bool = True
):
    rules_prompt = (
        f"Now, suggest {num_rules} rules that could be useful to make an optimal decision in the current state. "
        " For each rule, provide the explanation of why it is important to consider it at the given state."
        " Each rule should be in machine-readable JSON Lines format. Each line should follow the following schema:\n\n"
        # " {'background' str, 'rule': str, 'state relevance': str, 'goal relevance': str}\n\n"
        " {'background' str, 'rule': str, 'state relevance': str}\n\n"
        "- The 'background' should a brief introduction and motivation to the focus of the rule.\n"
        "- The 'rule' should be a statement of the form '[do/select/prioritize] [if/when/condition]' where the condition must be relevant to the current state.\n"
        "- The 'state relevance' should explain why the rule applies to the current problem state.\n"
        # "- The 'goal relevance' should explain why the rule is important to achieve the agent's goals.\n"
        "- The rule alone should be sufficient to deduce the optimal action that should be taken in the current problem state."
        "- Start each line with the character '```- {\"'.\n"
    )

    if example_rules is not None:
        rules_prompt += f"\n\n### Example rules\n\n{example_rules}\n\n"

    tmp_messages = messages.copy()
    tmp_messages.append({"role": "user", "content": rules_prompt})
    response = invoke_with_retries(llm, tmp_messages, max_tokens=512).content
    rules = parse_rules(response)
    outputs["rules"] = rules

    # send second call using the OpenAI API
    if save_prompts:
        messages.append({"role": "user", "content": rules_prompt})
        rules_str = "\n".join(outputs["rules"])
        messages.append({"role": "assistant", "content": rules_str})

    return rules


def _gen_rules_with_in_context_learning(
    outputs,
    messages,
    llm,
    num_rules,
    scored_rules: str,
    save_prompts: bool = True,
):
    rules_prompt = (
        f"Now, suggest {num_rules} rules that could be useful to make an optimal decision in the current state. "
        f"You will be given examples of rules ranked by their **fitness score** in [0,1]. Your goal is to propose only rules "
        " with high fitness scores.\n"
        "For each rule, provide the explanation of why it is important to consider it at the given state."
        " Each rule should be in machine-readable JSON Lines format. Each line should follow the following schema:\n\n"
        " {'background' str, 'rule': str, 'state relevance': str, 'goal relevance': str}\n\n"
        "- The 'background' should a brief introduction and motivation to the focus of the rule.\n"
        "- The 'rule' should be a statement of the form '[do/select/prioritize] [if/when/condition]' where the condition must be relevant to the current state.\n"
        "- The 'state relevance' should explain why the rule applies to the current problem state.\n"
        "- The 'goal relevance' should explain why the rule is important to achieve the agent's goals.\n"
        "- The rule alone should be sufficient to deduce the optimal action that should be taken in the current problem state.\n"
        "- You do not need to provide the fitness score, only the rule.\n"
        "- Start each line with the character '```- {\"'.\n"
        f"### Scored rules\n\n{scored_rules}\n\n"
    )

    tmp_messages = messages.copy()
    tmp_messages.append({"role": "user", "content": rules_prompt})
    response = invoke_with_retries(llm, tmp_messages, max_tokens=512).content
    rules = parse_rules(response)
    outputs["rules"] = rules

    # send second call using the OpenAI API
    if save_prompts:
        messages.append({"role": "user", "content": rules_prompt})
        rules_str = "\n".join(outputs["rules"])
        messages.append({"role": "assistant", "content": rules_str})

    return rules


def _gen_thoughts_with_in_context_learning(
    outputs,
    messages,
    llm,
    scored_thoughts: str,
    save_prompts: bool = True,
):
    prompt = (
        "Now, reason about what elements should be considered when choosing the optimal action."
        " Your response should consist of a single short paragraph that reflects on the consequences, benefits, and drawbacks"
        " of each action in the current state."
        f"Below are examples of answers ranked by their **quality score** in [0,1]. ## Example answers\n\n{scored_thoughts}\n\n"
    )
    tmp_messages = messages.copy()
    tmp_messages.append({"role": "user", "content": prompt})
    response = invoke_with_retries(llm, tmp_messages, temperature=0.5, max_tokens=256).content

    if save_prompts:
        outputs["thoughts"] = response
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": outputs["thoughts"]})

    return response


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
        use_thoughts: bool = True,
    ):
        super().__init__(
            task_text=task_text,
            action_space_text=action_space_text,
            llm=llm,
            use_thoughts=use_thoughts,
        )
        self.num_rules = num_rules
        self.example_rules = example_rules
        self.max_parse_attempts = max_parse_attempts
        self.verbose = verbose

    def pre_action(self, outputs: Dict, messages: List[Dict]):
        super().pre_action(outputs, messages)
        self.gen_rules(outputs, messages)

    def post_action(self, outputs: Dict, messages: List[Dict]):
        super().post_action(outputs, messages)
        self.gen_rule_scores(outputs, messages)

    def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
        return _gen_thoughts_for_rule_agents(outputs, messages, self.llm)

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        # get actions
        action_prompt = (
            "Now, choose the optimal action given the current problem state and the chosen priorization rules. "
            "Your answer must consist exclusively of one of the following actions:"
            f"\n\n### Possible actions:\n\n{self.action_space_text}"
            "\n\nYou cannot refuse to respond. Do not provide additional information or context for your answer, only the action."
        )
        messages.append({"role": "user", "content": action_prompt})

        outputs["action"] = invoke_with_retries(
            self.llm, messages, max_tokens=30
        ).content
        messages.append({"role": "assistant", "content": outputs["action"]})

        return outputs["action"]

    def gen_rules(self, outputs: Dict, messages: List[Dict]):
        _gen_rules(outputs, messages, self.llm, self.num_rules, self.example_rules)

    def gen_rule_scores(self, outputs: Dict, messages: List[Dict]):
        system_prompt = self.system_prompt_with_state(outputs["state_text"])
        rules = outputs["rules"]
        return _gen_rule_scores(outputs, messages, self.llm, rules, system_prompt)


LLMRulesAgentNoThoughts = partial(LLMRulesAgent, use_thoughts=False)


class RulesSelectorActorCritic(BaseAgent):
    """The rule-based agent generates a set of rules based on the environment state."""

    def __init__(
        self,
        actor: layers.CrossAttentionNetwork,
        task_text: str,
        action_space_text: str,
        llm: BaseChatModel,
        embededder: Embeddings,
        max_rule_combinations: int = 3,
        num_rules: int = 5,
        example_rules: Optional[str] = None,
        max_parse_attempts: int = 3,
        verbose: bool = False,
        critic: Optional[layers.CrossAttentionNetwork] = None,
        deterministic: bool = False,
        use_thoughts: bool = True,
        in_context_learning: bool = False,
        optimize_thoughts_only: bool = False,
    ):
        super().__init__(
            task_text=task_text,
            action_space_text=action_space_text,
            llm=llm,
            use_thoughts=use_thoughts,
        )

        self.in_context_learning = in_context_learning
        self.actor = actor
        self.critic = critic
        self.max_rule_combinations = max_rule_combinations
        self.embedder = embededder
        self.num_rules = num_rules
        self.example_rules = example_rules
        self.max_parse_attempts = max_parse_attempts
        self.verbose = verbose
        self.deterministic = deterministic
        self.optimize_thoughts_only = optimize_thoughts_only
        if self.optimize_thoughts_only:
            self.use_thoughts = False # they will be randomly generated

    def pre_action(self, outputs: Dict, messages: List[Dict]):
        super().pre_action(outputs, messages)
        self.gen_rules(outputs, messages)

    def gen_rules(self, outputs: dict, messages: list[dict]) -> List[str]:
        """Wrapper for generating rules that includes combinations of and embeddings for rules.
        First, generates combinations of rules, next it filters using an RL selector
        """
        max_attemps = 0
        while max_attemps < self.max_parse_attempts:
            try:
                if not self.optimize_thoughts_only:
                    rules = _gen_rules(
                        outputs,
                        messages,
                        self.llm,
                        self.num_rules,
                        self.example_rules,
                        save_prompts=False,
                    )
                else:
                    rules = [
                        _gen_thoughts_for_rule_agents(outputs, messages, self.llm, save_prompts=False)
                        for _ in range(self.num_rules)
                    ]
                    outputs["rules"] = rules
                # check that we have at least one rule to select from
                if isinstance(rules, list) and len(rules) > 0:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"Error: {e}")
                max_attemps += 1
        if max_attemps == self.max_parse_attempts:
            raise ValueError("Failed to generate rules")
        
        if self.in_context_learning:
            # use the critic to rank the rules
            rules_emb = self.embedder.embed_documents(rules)
            dev = next(self.actor.parameters()).device
            rules_emb = torch.tensor(rules_emb, dtype=torch.float32).to(dev)
            state_vector = outputs["state_vector"]
            if state_vector.dim() == 1:
                state_vector = state_vector.unsqueeze(0)

            queries, keys = rules_emb, state_vector
            if rules_emb.shape[0] > 1:
                with torch.no_grad():
                    values = self.critic(queries, keys).squeeze(0).cpu().detach().numpy()
                    # values = (values - values.mean()) / (values.std() + 1e-6)
                    values = 0.1 + 0.8 * (values - values.min()) / (values.max() - values.min())

                # append the the score to each rule
                scored_rules = [
                    f"{r} --> {{'score': {v.item():.2f}}}" for r, v in zip(rules, values)
                ]
                outputs["scored_rules"] = scored_rules

                # sort the rules by the critic values
                ix = np.argsort(values)[::-1]
                scored_rules = [scored_rules[i] for i in ix]

                if not self.optimize_thoughts_only:
                    new_rules = _gen_rules_with_in_context_learning(
                        outputs,
                        messages,
                        self.llm,
                        self.num_rules,
                        scored_rules,
                        save_prompts=False,
                    )
                else:
                    # here we save the prompt/answerbecause we will use them to optimize the thoughts
                    new_rules = [
                        _gen_thoughts_with_in_context_learning(
                            outputs, messages, self.llm, scored_rules, save_prompts=True
                        )
                        for _ in range(self.num_rules)
                    ]
                rules = new_rules

        # rules = generate_rule_combinations(. #TODO: implement this
        #     rules, max_combs=self.max_rule_combinations
        # )

        outputs["rules"] = rules

        # dont' add all rules, confuses the LLM and increases the cost
        # messages.append({"role": "assistant", "content": rules_str})

        # get the rules and state embeddings
        device = next(self.actor.parameters()).device
        rules_emb = self.embedder.embed_documents(rules)
        rules_emb = torch.tensor(rules_emb, dtype=torch.float32).to(device)
        outputs["rules_emb"] = rules_emb

        # get the rule scores
        state_vector = outputs["state_vector"]
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)

        queries, keys = rules_emb, state_vector
        with torch.no_grad():
            logits = self.actor(queries, keys)

        dist = torch.distributions.Categorical(logits=logits)
        if not self.deterministic:
            sel_idx = dist.sample()
        else:
            sel_idx = torch.argmax(logits)

        entropy = dist.entropy()

        # get the selected rule
        sel_rule = rules[sel_idx]

        outputs["logits"] = logits
        outputs["sel_logprob"] = dist.log_prob(sel_idx)
        outputs["sel_idx"] = sel_idx
        outputs["sel_rule"] = sel_rule
        outputs["entropy"] = entropy

        if hasattr(self, "critic") and self.critic is not None:
            value = self.critic(queries, keys)
            outputs["value"] = value.squeeze()

    def gen_rule_scores(self, outputs: Dict, messages: List[Dict]):
        system_prompt = self.system_prompt_with_state(outputs["state_text"])
        sel_rule = outputs["sel_rule"]
        return _gen_rule_scores(outputs, messages, self.llm, [sel_rule], system_prompt)

    def post_action(self, outputs, messages):
        super().post_action(outputs, messages)
        self.gen_rule_scores(outputs, messages)

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        # get actions
        action_prompt = (
            f"Below is/are a priorization rule/rules to make an optimal decision in the current state:\n\n"
            f"{outputs['sel_rule']}\n\n"
            "\n\n"
            "Now, choose the optimal action given the current problem state and this/these priorization rule/rules. "
            "Your answer must consist exclusively of one of the following actions:"
            f"\n\n### Possible actions:\n\n{self.action_space_text}"
            "\n\nYou cannot refuse to respond. Do not provide additional information or context for your answer, only the action."
        )
        messages.append({"role": "user", "content": action_prompt})

        outputs["action"] = invoke_with_retries(
            self.llm, messages, max_tokens=30
        ).content
        messages.append({"role": "assistant", "content": outputs["action"]})
        return outputs["action"]

    def get_action_and_value_from_embeddings(
        self,
        state_vector: torch.Tensor,
        rules_emb: torch.Tensor,
        rules_padding_mask: Optional[torch.Tensor] = None,
        sel_idxs: Optional[torch.Tensor] = None,
    ):
        queries, keys = rules_emb, state_vector
        logits = self.actor(queries, keys, key_padding_mask=rules_padding_mask)

        dist = torch.distributions.Categorical(logits=logits)
        if sel_idxs is None:
            sel_idxs = dist.sample()
        entropy = dist.entropy()
        log_prob = dist.log_prob(sel_idxs)

        values = self.critic(
            state_vector.unsqueeze(1), rules_emb, key_padding_mask=rules_padding_mask
        )

        return sel_idxs, log_prob, entropy, values

    def get_policy_from_embeddings(
        self,
        state_vector: torch.Tensor,
        rules_emb: torch.Tensor,
        rules_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        if state_vector.dim() == 2:
            state_vector = state_vector.unsqueeze(1)
        queries, keys = rules_emb, state_vector
        logits = self.actor(queries, keys, key_padding_mask=rules_padding_mask)

        if logits.is_nested:
            # pad them with a negative number
            logits = torch.nested.to_padded_tensor(logits, -100.0)

        dist = torch.distributions.Categorical(logits=logits)

        return dist

    def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
        return _gen_thoughts_for_rule_agents(outputs, messages, self.llm)


class LLMFineTuningAgent(BaseAgent):
    """This agents uses a similar pipeline than the base agent. A main difference is that
    the action generation uses the LLM as a tranformer object that can be fined-tuned using
    HuggingFace tools.

    For speed, the thoughts and explanation are still generated using the base agent pipeline.
    """

    import transformers

    def __init__(
        self,
        task_text: str,
        action_space_text: str,
        llm: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        wrapped_llm = HFMetaWrapper(llm, tokenizer)
        super().__init__(
            task_text=task_text,
            action_space_text=action_space_text,
            llm=wrapped_llm,
        )
        self.tokenizer = tokenizer

        # store input llm as 'network'
        self.network = llm