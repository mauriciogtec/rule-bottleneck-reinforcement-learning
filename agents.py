from collections import deque
from functools import partial
import json
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
    patterns = [
        "```yaml",
        "```yml",
        "```",
        "```json",
        "json",
    ]
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
            messages = [{"role": "user", "content": initial_prompt}]

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
            f"You are an agent tasked solving a sequential decision making problem."
            " You will be given the task context, problem state, and possible actions."
            " Your goal is to choose the optimal action to maximize the future cumulative reward."
            f"\n\n### Problem\n\n{self.task_text}"
            f"\n\n### Current state of the  problem\n\n{state_text}"
            f"\n\n### Possible actions\n\n{self.action_space_text}"
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
            self.llm, messages, max_tokens=10, temperature=0.2
        ).content
        messages.append({"role": "assistant", "content": outputs["action"]})

        # outputs["action"] = self.action_parser(outputs["action_str"])
        return outputs["action"]

    def post_action(self, outputs: Dict, messages: List[Dict]):
        """Finalizes outputs and messages"""
        self.gen_explanation(outputs, messages)

    def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
        return _gen_thoughts(outputs, messages, self.llm)

    def gen_explanation(self, outputs: Dict, messages: List[Dict]):
        """Generate explanation and update message list"""
        _gen_explanation(outputs, messages, self.llm)
        # messages.append({"role": "assistant", "content": outputs["explanation"]})


NoThoughtsAgent = partial(BaseAgent, use_thoughts=False)


def _gen_thoughts(outputs, messages, llm, save_prompts: bool = True):
    thoughts_prompt = (
        "Now, let's take it step by step. First, reason about what elements should be"
        " considered when choosing the optimal action considering that actios affect the future state of the system."
        " Thoughts must contain: 1) observations about the current state of the problem;",
        " 2) observations about the reward/cost; 3) how each action will affect the future state of the system;"
        " 4) other believes about the future state of the system."
        " Do not make more than 5 thoughts and each thought should be at most 6 words."
    )
    system_prompt = outputs["initial_prompt"]    
    tmp_messages = [{"role": "user", "content": system_prompt + f"\n\n###Thoughts task\n\np{thoughts_prompt}"}]
    thoughts = invoke_with_retries(
        llm, tmp_messages, temperature=0.5, max_tokens=256
    ).content
    if save_prompts:
        # messages.append({"role": "user", "content": prompt})
        # messages.append({"role": "assistant", "content": outputs["thoughts"]})
        # save the prompt and response
        # messages.append({"role": "user", "content": outputs["initial_prompt"] + prompt})
        outputs["thoughts"] = thoughts
    # messages.append({"role": "assistant", "content": outputs["thoughts"]})
    return thoughts


def _gen_rule_scores(outputs, messages, llm, rules, system_prompt):
    """Generate rule scores based on the current state and the set of rules."""
    # Get only the task prompt with state and prompt the LLM to check whether the
    # selected rule is enough to understand
    # 1) What will the agent to next (action prediction)?
    # 2) Is the rule relevant to the current state?
    # 3) Is the justification of the rule clear and relates to the task?

    q1 = "Is/are the rule/rules sufficient to predict the exact action that the system will take in the current state?"
    q2 = "Is the background motivation of the rule hallucinations free and relevant to the current state?"

    q3 = "Did the rule/rules predicted the optimal action/decision that the system took (answer no=0 if contradiction)?"
    q4 = "Was the background motivation for the rule hallucinations free and relevant to the current state?"

    rules = "\n".join(rules)
    rule_scores_prompt = (
        "### Rule evaluation task"
        "Evaluate the usefulness of the following rule to determine the optimal action:\n\n"
        f"\n\n{rules}\n\n"
        "You will need to provde a simple answer 'yes' or 'no' to following questions.\n\n"
        "### Questions\n\n"
        "1. {q1}\n"
        "2. {q2}\n"
        "Answer in JSON format where (no=0, yes=1): For example: {'q1': 0, 'q2': 0}\n\n"
        ""
    )
    post_hoc_prompt = (
        "### Rule evaluation task\n\n"
        "Evaluate the compatibility of the following rule with the current state and the selected action.\n\n"
        f"### Selected action\n\n{outputs['action']}\n\n"
        f"### Selected rule(s)\n\n{rules}\n\n"
        "### Questions"
        "1. {q3}\n"
        "2. {q4}\n"
        "Answer in JSON format where (no=0, yes=1). For example: {'q1': 0, 'q2': 0}\n\n"
    )

    # q1 = "Q1. Is/are the rule/rules complete, i.e., sufficient to determine the optimal action/decision that the system will take in current the problem state?"
    # # q2 = "Q2. Is the ful"
    # q2 = "Q2. Is the justification of the rule satisfactory without false logic or hallucinations?"
    # q3 = "Q3. Did the selected rule/rules sufficiently help to understand the previous decision without contradictions?"

    # coda = (
    #     "\nAnswer the following questions with a simple 'yes' or 'no' without additional"
    #     " information or justification. Your response should be a single word.\n\n"
    # )

    # Answer rules prompt
    # temp_messages = messages.copy()
    tmp_messages = [{"role": "user", "content": system_prompt + rule_scores_prompt}]
    response1 = invoke_with_retries(
        llm, tmp_messages, max_tokens=20, temperature=0.
    ).content
    # use regex to extract the values
    try:
        # extract braces content , e.g., "I choose ```json{'q1': 1, 'q2': 0}```" -> "{'q1': 1, 'q2': 0}"
        numbers = response1.split('{')[1].split('}')[0]
        numbers = json.loads("{" + numbers + "}")
        r1, r2 = [float(numbers[k]) for k in numbers.keys()]
    except:
        r1, r2 = 0, 0

    # Answer post hoc prompt
    tmp_messages = [{"role": "user", "content": system_prompt + post_hoc_prompt}]
    response2 = invoke_with_retries(
        llm, tmp_messages, max_tokens=20, temperature=0.2
    ).content
    # use regex to extract the values
    try:
        # firest find content between braces the parson with json.loads
        numbers = response2.split("{")[1].split("}")[0]
        numbers = json.loads("{" + numbers + "}")
        r3, r4 = [float(numbers[k]) for k in numbers.keys()]
    except:
        r3, r4 = 0, 0

    # msg = rule_scores_prompt + "### Question\n\n" + q1 + coda
    # temp_messages.append({"role": "user", "content": msg})
    # r1_ = invoke_with_retries(llm, temp_messages, max_tokens=2).content
    # r1 = float("yes" in r1_.lower())
    # temp_messages.append({"role": "assistant", "content": r1_})

    # # Answer q2
    # temp_messages = messages.copy()
    # msg = rule_scores_prompt + "### Question\n\n" + q2 + coda
    # temp_messages.append({"role": "user", "content": msg})
    # r2_ = invoke_with_retries(llm, temp_messages, max_tokens=2).content
    # r2 = float("yes" in r2_.lower())
    # temp_messages.append({"role": "assistant", "content": r2_})

    # # Answer q3
    # temp_messages = messages.copy()
    # msg = rule_scores_prompt + (
    #     f"The decision taken in the current problem state was: {outputs['action']}.\n\n"
    #     f"### Question\n\n{q3 + coda}"
    # )
    # temp_messages.append({"role": "user", "content": msg})
    # r3_ = invoke_with_retries(llm, temp_messages, max_tokens=2).content
    # r3 = float("yes" in r3_.lower())
    # temp_messages.append({"role": "assistant", "content": r3_})

    # # Answer q4
    # temp_messages = messages.copy()
    # msg = rule_scores_prompt + "### Question\n\n" + q4 + coda
    # temp_messages.append({"role": "user", "content": msg})
    # r4_ = invoke_with_retries(llm, temp_messages, max_tokens=2).content
    # r4 = float("yes" in r4_.lower())
    # temp_messages.append({"role": "assistant", "content": r4_})

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
    outputs["sel_reward_scores_raw"] = {q1: response1, q2: response1, q3: response2, q4: response2}


# def _gen_thoughts_for_rule_agents(outputs, messages, llm, save_prompts: bool = True):
#     # prompt = (
#     #     "First, reason about what elements should be considered when choosing the optimal action"
#     #     " in the given task of the decision making agent."
#     #     " Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks"
#     #     " of each action in the current state. Conclude the paragraph with a reflection of how they inform the design"
#     #     " of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
#     # )
#     prompt = (
#         "First, reason about what elements should be considered when choosing the optimal action."
#         " Your response should consist of a single short paragraph that reflects on the consequences, benefits, and drawbacks"
#         " of each action in the current state."
#     )
#     tmp_messages = messages.copy()
#     tmp_messages.append({"role": "user", "content": prompt})
#     response = invoke_with_retries(
#         llm, tmp_messages, temperature=0.5, max_tokens=256
#     ).content

#     if save_prompts:
#         outputs["thoughts"] = response
#         messages.append({"role": "user", "content": prompt})
#         messages.append({"role": "assistant", "content": outputs["thoughts"]})

#     return response

def _gen_explanation(outputs, messages, llm, use_thoughts=True):
    explanation_prompt = ""

    if use_thoughts:
        explanation_prompt += (
            f"### Thoughts\n\n"
            f"Given the problem state, below are your previous thoughts used to make a decision\n: {outputs['thoughts']}\n\n"
        )

    explanation_prompt += (
        f"### Selected Action\n\n"
        "Given the problem state and your previous reasoning, you selected action(s) to make a decision\n:"
        f"{outputs['action']}\n\n"
        f"### Explanation Task\n\n"
        "Explain how you chose the optimal action."
        " Your response should be a short paragraph."
        " Your explanation is meant for a human audience which cannot see the previous reasoning, so it must be self-contained."
        " Do not guess. Answer solely based on the problem state and and previous reasoning about expectations of each action, sequential dependence, and reasoning about future reward/cost, ."
        " Follow the template:"
        " In the current state, I observed <state>. "
        " Then, I reasoned that <thoughts>."
        " I concluded that <action> is the optimal action.\n"
        " - For the state and thoughts, include only the facts of the state that you used later to make a decision\n"
        # " - For the thoughts, include expectations of each action, reasoning about future reward/cost, include only the most important thoughts. \n"
        " - In both cases, use sentences with less than six words and no more than 3 sentences.\n"
    )

    tmp_messages = [
        {"role": "user", "content": outputs["initial_prompt"] + explanation_prompt}
    ]
    outputs["explanation"] = invoke_with_retries(
        llm, tmp_messages, temperature=0.2, max_tokens=200
    ).content

def _gen_explanation_rules(outputs, messages, llm, use_thoughts=True):

    explanation_prompt = outputs["initial_prompt"] + "\n\n"

    if use_thoughts:
        explanation_prompt += (
            f"### Thoughts\n\n"
            f"Given the problem state, below are your previous thoughts used to make a decision\n: {outputs['thoughts']}\n\n"
        )

    # add rules
    if "sel_rule" in outputs:
        explanation_prompt += (
            f"\n\n### Selected rule\n\n"
            f"Given your previous reasoning, you selected the following rule to make a decision\n:{outputs['sel_rule']}\n\n"
        )
    else:
        explanation_prompt += (
            f"\n\n### Selected rule\n\n"
            f"Given your previous reasoning, you selected the following rules to make a decision\n:{outputs['rules']}\n\n"
        )

    explanation_prompt += (
        f"### Selected Action\n\n"
        "Given the problem state and your previous reasoning, you selected action(s) to make a decision\n:"
        f"{outputs['action']}\n\n"
        f"### Explanation Task\n\n"
        "Explain how you chose the optimal action."
        " Your response should be a short paragraph."
        " Your explanation is meant for a human audience which cannot see the previous reasoning, so it must be self-contained."
        " Do not guess. Answer solely based on the problem state and and previous reasoning about expectations of each action, sequential dependence, and reasoning about future reward/cost, ."
        " Follow the template:"
        " In the current state, I observed <state>. "
        " Then, I reasoned that <rule motivation in concise natural language>."
        " I applied the rule stating that <rule recipe>."
        " Applying this rule to current state, I concluded that <action> is the optimal action.\n"
        " - For the state, include only the facts of the state that you used later to make a decision\n"
        # " - For the thoughts, include expectations of each action, reasoning about future reward/cost, include only the most important thoughts. \n"
        " - In both cases, use sentences with less than six words and no more than 3 sentences.\n"
    )

    tmp_messages = [{"role": "user", "content": explanation_prompt}]
    outputs["explanation"] = invoke_with_retries(
        llm, tmp_messages, temperature=0.2, max_tokens=200
    ).content
    # messages.append({"role": "assistant", "content": outputs["explanation"]})


def _gen_rules(
    outputs, messages, llm, num_rules=5, example_rules=None, save_prompts: bool = True, 
):
    
    rules_prompt = outputs["initial_prompt"] 
    if "thoughts" in outputs:
        rules_prompt += (
            f"\n\n### Thoughts\n\n"
            f"Given the problem state, below are your previous thoughts used to make a decision\n: {outputs['thoughts']}\n\n"
        )

    if num_rules > 1:
        rules_prompt += (
            f"\n\n### Rule generation task\n\n"
            f"Now, suggest a set of {num_rules} potential rules that could be applied to find the optimal decision in the current state.\n"
            "There must be diversity among the candidate rules.\n"
            "When the optimal action is not fully certain, it is better to suggest diverse rules leading to different actions.\n"
        )
    else:
        rules_prompt += (
            f"\n\n### Rule generation task\n\n"
            f"Now, suggest a rule that can be applied to make optimal decision in the current state."
        )

    rules_prompt += (
        " Provide one line per rule in the following JSON schema:\n\n"
        # " {'background' str, 'rule': str, 'state relevance': str, 'goal relevance': str}\n\n"
        # " {'background' <str>, 'rule': <str>, 'state relevance': <str>}\n\n"
        " {'background' <str>, 'rule': <str>}\n\n"
        # " - {'rule': str, 'background': str}\n\n"
        "- The 'background' should be a rationale to determine an optimal action explicitly reasoning about future"
        " the sequential nature of the problem and how each action would allow maximizing cumulative reward/minimizing cost."
        " It must consist of up to 4 sentences of at most 6 words each.\n"
        "- The rule should describe an explicit recipe to determine the optimal action as a function of the current problem state\n"
        "- A rule must be reusable in different states, so it should contain specific values of the problem state. But a recipe instead"
        " However, the rule must allow to determine the optimal action in the current state.\n"
        # "- The 'state relevance' should explain why the rule applies to the current problem state.\n"
        # "- The 'goal relevance' should explain why the rule is important to achieve the agent's goals.\n"
        # "- The rule alone should be sufficient to deduce the optimal action that should be taken in the current problem state."
    )

    # if num_rules > 1:
    #     rules_prompt += "- Rules should be self-contained and not depend on other rules. The best rule will be selected later.\n"
    
    rules_prompt += "- Each line of the response should start with the characters '```- {\"'.\n"

    if example_rules is not None:
        rules_prompt += (
            f"\n\n### Example rules\n\n"
            f"\n\n{example_rules}\n\n"
        )
    

    tmp_messages = [{"role": "user", "content": rules_prompt}]
    response = invoke_with_retries(llm, tmp_messages, max_tokens=1024, temperature=2.0).content
    rules = parse_rules(response)
    outputs["rules"] = rules
    outputs["rules_str"] = response

    # send second call using the OpenAI API
    if save_prompts:
        # messages.append({"role": "user", "content": rules_prompt})
        rules_str = "\n".join(outputs["rules"])
        # messages.append({"role": "assistant", "content": rules_str})
        # outputs["rules"] = rules_str

    return rules


def _gen_rules_with_in_context_learning(
    outputs,
    messages,
    llm,
    num_rules,
    scored_rules: str,
    save_prompts: bool = True,
):
    rules_prompt = outputs["initial_prompt"]

    if "thoughts" in outputs:
        rules_prompt += (
            f"\n\n### Thoughts\n\n"
            f"Given the problem state, below are your previous thoughts used to make a decision\n: {outputs['thoughts']}\n\n"
        )

    rules_prompt += (
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


    tmp_messages = [{"role": "user", "content": rules_prompt}]
    response = invoke_with_retries(llm, tmp_messages, max_tokens=512).content
    rules = parse_rules(response)
    outputs["rules"] = rules

    # send second call using the OpenAI API
    if save_prompts:
        # messages.append({"role": "user", "content": rules_prompt})
        rules_str = "\n".join(outputs["rules"])
        # messages.append({"role": "assistant", "content": rules_str})
        # outputs["rules"] = rules_str

    return rules


def _gen_thoughts_with_in_context_learning(
    outputs,
    messages,
    llm,
    scored_thoughts: str,
    save_prompts: bool = True,
):
    thoughts_prompt = (
        "Now, let's take it step by step. First, reason about what elements should be"
        " considered when choosing the optimal action considering that actios affect the future state of the system."
        " Thoughts must content (1) observations about the current state of the problem,",
        " (2) observations about the reward/cost (3) reasoning about the how each action"
        " will affect the future state of the system; (4) other believes about the future state of the system."
        " Do not make more than 5 thoughts and each thought should be at most 6 words.\n\n"
        f"Below are examples of answers ranked by their **quality score** in [0,1]. Your goal is to propose thoughts with high quality scores.\n\n"
        f"### Example answers\n\n{scored_thoughts}\n\n"
    )
    # tmp_messages = messages.copy()
    tmp_messages = [{"role": "user", "content": thoughts_prompt + "### Thoughts task\n\n"}]
    response = invoke_with_retries(
        llm, tmp_messages, temperature=0.9, max_tokens=256
    ).content

    if save_prompts:
        outputs["thoughts"] = response
        # messages.append({"role": "user", "content": prompt})
        # messages.append({"role": "assistant", "content": outputs["thoughts"]})

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

    # def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
    #     return _gen_thoughts_for_rule_agents(outputs, messages, self.llm)

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        # get actions
        action_prompt = outputs["initial_prompt"] + "\n\n"

        if "thoughts" in outputs:
            action_prompt += (
                f"### Thoughts\n\n"
                f"Given the problem state, below are your previous thoughts used to make a decision\n: {outputs['thoughts']}\n\n"
            )
        
        action_prompt += (
            f"\n\n### Selected rules\n\n"
            f"Given your previous reasoning, you selected the following rules to make a decision\n:{outputs['rules']}\n\n"
        )

        action_prompt += (
            "### Action selection task\n\n"
            "Now, choose the optimal action based only on the selected rule and the current problem state. "
            f"\n\n### Possible actions:\n\n{self.action_space_text}"
            "\n\nYou cannot refuse to respond. Do not provide additional information or context for your answer, only the action.\n"
            "Answer in JSON format. For example: {'action': 0}."
        )
        # messages.append({"role": "user", "content": action_prompt})

        tmp_messages = [{"role": "user", "content": action_prompt}]
        outputs["action"] = invoke_with_retries(
            self.llm, tmp_messages, max_tokens=30, temperature=0.2
        ).content
        messages.append({"role": "assistant", "content": outputs["action"]})

        return outputs["action"]

    def gen_rules(self, outputs: Dict, messages: List[Dict]):
        _gen_rules(outputs, messages, self.llm, self.num_rules, self.example_rules)

    def gen_rule_scores(self, outputs: Dict, messages: List[Dict]):
        system_prompt = self.system_prompt_with_state(outputs["state_text"])
        rules = outputs["rules"]
        return _gen_rule_scores(outputs, messages, self.llm, rules, system_prompt)
    
    def gen_explanation(self, outputs: Dict, messages: List[Dict]):
        """Generate explanation and update message list"""
        _gen_explanation_rules(outputs, messages, self.llm, use_thoughts=self.use_thoughts)


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
        max_rule_combinations: int = 1,
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
            self.use_thoughts = False  # they will be randomly generated

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
                        _gen_thoughts(
                            outputs, messages, self.llm, save_prompts=False
                        )
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
                    values = (
                        self.critic(queries, keys).squeeze(0).cpu().detach().numpy()
                    )
                    # values = (values - values.mean()) / (values.std() + 1e-6)
                    values = 0.1 + 0.8 * (values - values.min()) / (
                        values.max() - values.min()
                    )

                # append the the score to each rule
                scored_rules = [
                    f"{r} --> {{'score': {v.item():.2f}}}"
                    for r, v in zip(rules, values)
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
        # action_prompt = (
        #     f"Below is/are a priorization rule/rules to make an optimal decision in the current state:\n\n"
        #     f"{outputs['sel_rule']}\n\n"
        #     "\n\n"
        #     "Now, choose the optimal action given the current problem state and this/these priorization rule/rules. "
        #     "Your answer must consist exclusively of one of the following actions:"
        #     f"\n\n### Possible actions:\n\n{self.action_space_text}"
        #     "\n\nYou cannot refuse to respond. Do not provide additional information or context for your answer, only the action."
        # )
        # messages.append({"role": "user", "content": action_prompt})
        # get actions
        action_prompt = outputs["initial_prompt"] + "\n\n"

        if "thoughts" in outputs:
            action_prompt += (
                f"### Thoughts\n\n"
                f"Given the problem state, below are your previous thoughts used to make a decision\n: {outputs['thoughts']}\n\n"
            )
        action_prompt += (
            f"\n\n### Selected rule\n\n"
            f"Given your previous reasoning, you selected the following rule to make a decision\n:{outputs['sel_rule']}\n\n"
        )

        action_prompt += (
            "### Action selection task\n\n"
            "Now, choose the optimal action given the selected rule and the current problem state. "
            f"\n\n### Possible actions:\n\n{self.action_space_text}"
            "\n\nYou cannot refuse to respond. Do not provide additional information or context for your answer, only the action."
            "Answer in JSON format. For example: {'action': 0}."
        )
        # messages.append({"role": "user", "content": action_prompt})

        tmp_messages = [{"role": "user", "content": action_prompt}]

        outputs["action"] = invoke_with_retries(
            self.llm, tmp_messages, max_tokens=30, temperature=0.2
        ).content

        # messages.append({"role": "assistant", "content": outputs["action"]})

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

    # def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
    #     return _gen_thoughts_for_rule_agents(outputs, messages, self.llm)

    def gen_explanation(self, outputs: Dict, messages: List[Dict]):
        """Generate explanation and update message list"""
        if self.optimize_thoughts_only:
            _gen_explanation(outputs, messages, self.llm, use_thoughts=self.use_thoughts)
        else:
            _gen_explanation_rules(outputs, messages, self.llm, use_thoughts=self.use_thoughts)


class RulesSelectorActorCriticRAG(BaseAgent):
    """The rule-based agent generates a set of rules based on the environment state."""

    def __init__(
        self,
        actor: layers.CrossAttentionNetwork,
        task_text: str,
        action_space_text: str,
        llm: BaseChatModel,
        embedder: Embeddings,
        max_rule_combinations: int = 1,
        num_gen_rules: int = 1,
        store_size: int = 1000,
        num_store_queries: int = 20,
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
        self.embedder = embedder
        self.num_store_queries = num_store_queries  # number of rules to select from
        self.num_gen_rules = num_gen_rules  # number of rules to generate
        self.example_rules = example_rules
        self.max_parse_attempts = max_parse_attempts
        self.verbose = verbose
        self.deterministic = deterministic
        self.optimize_thoughts_only = optimize_thoughts_only
        self.rule_store = deque(maxlen=store_size)
        if self.optimize_thoughts_only:
            self.use_thoughts = False  # they will be randomly generated

    def similarity_search(self, query: torch.Tensor) -> list[dict]:
        """
        Perform vector similarity search over a store of embeddings.

        Each store entry is a dictionary with the keys:
        - "key": the vector (torch.Tensor)
        - "metadata": additional data associated with the vector.

        Parameters:
        - query (torch.Tensor): The query vector.

        Returns:
        - list[dict]: The top_k items from the store with the highest similarity.
        """

        if len(self.rule_store) == 0:
            return []

        # Convert the stored vectors to a single torch tensor.
        vectors = torch.stack(
            [
                (
                    item["key"]
                    if isinstance(item["key"], torch.Tensor)
                    else torch.tensor(item["key"])
                )
                for item in self.rule_store
            ]
        )

        # Normalize the query and stored vectors.
        query_norm = query / torch.norm(query)
        vectors_norm = vectors / torch.norm(vectors, dim=1, keepdim=True)
        # Compute cosine similarity as dot product.
        scores = torch.matmul(vectors_norm, query_norm)
        # Higher score means higher similarity.
        top_indices = torch.argsort(scores, descending=True)[
            : (self.num_store_queries - 1)
        ]
        # Return the top_k results from the store.
        return [self.rule_store[i] for i in top_indices.tolist()]

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
                        self.num_gen_rules,
                        self.example_rules,
                        save_prompts=False,
                    )
                else:
                    rules = [
                        _gen_thoughts(
                            outputs, messages, self.llm, save_prompts=False
                        )
                        for _ in range(self.num_gen_rules)
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
                    values = (
                        self.critic(queries, keys).squeeze(0).cpu().detach().numpy()
                    )
                    # values = (values - values.mean()) / (values.std() + 1e-6)
                    values = 0.1 + 0.8 * (values - values.min()) / (
                        values.max() - values.min()
                    )

                # append the the score to each rule
                scored_rules = [
                    f"{r} --> {{'score': {v.item():.2f}}}"
                    for r, v in zip(rules, values)
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

        outputs["rules"] = rules

        # get the rules and state embeddings
        device = next(self.actor.parameters()).device
        rules_emb = self.embedder.embed_documents(rules)
        rules_emb = torch.tensor(rules_emb, dtype=torch.float32).to(device)
        outputs["rules_emb"] = rules_emb

        # new here!! --- Need to retrieve rules from database
        # we will save the state as the document and the rules and their embeddings as the metadata
        retrieved = self.similarity_search(outputs["state_vector"])
        if len(retrieved) > 0:
            # we have some retrieved rules
            retrieved_rules = [item["metadata"]["rules"] for item in retrieved]
            retrieved_rule_embs = [item["metadata"]["rule_embs"] for item in retrieved]

            # concatenate the new rules with the retrieved ones
            rules += retrieved_rules
            rules_emb = torch.cat(
                [rules_emb, torch.stack(retrieved_rule_embs)],
            )

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

        # store the state vector and the selected rule in the rule store
        for j in range(self.num_gen_rules):
            rule_entry = {
                "key": outputs["state_vector"],
                "metadata": {
                    "rules": outputs["rules"][j],
                    "rule_embs": rules_emb[j],
                    "state_text": outputs["state_text"],
                },
            }
            self.rule_store.append(rule_entry)

        # updat ethe output rules to include all the rules
        outputs["rules"] = rules
        outputs["rules_emb"] = rules_emb

    def gen_rule_scores(self, outputs: Dict, messages: List[Dict]):
        system_prompt = self.system_prompt_with_state(outputs["state_text"])
        sel_rule = outputs["sel_rule"]
        return _gen_rule_scores(outputs, messages, self.llm, [sel_rule], system_prompt)

    def post_action(self, outputs, messages):
        super().post_action(outputs, messages)
        self.gen_rule_scores(outputs, messages)

    def get_action(self, outputs: Dict, messages: List[Dict]) -> ActType:
        action_prompt = outputs["initial_prompt"] + "\n\n"

        if "thoughts" in outputs:
            action_prompt += (
                f"### Thoughts\n\n"
                f"Given the problem state, below are your previous thoughts used to make a decision\n: {outputs['thoughts']}\n\n"
            )
        action_prompt += (
            f"\n\n### Selected rule\n\n"
            f"Given your previous reasoning, you selected the following rule to make a decision\n:{outputs['sel_rule']}\n\n"
        )

        action_prompt += (
            "### Action selection task\n\n"
            "Now, choose the optimal action given the selected rule and the current problem state. "
            f"\n\n### Possible actions:\n\n{self.action_space_text}"
            "\n\nYou cannot refuse to respond. Do not provide additional information or context for your answer, only the action."
            "Answer in JSON format. For example: {'action': 0}."
        )
        # messages.append({"role": "user", "content": action_prompt})

        tmp_messages = [{"role": "user", "content": action_prompt}]

        outputs["action"] = invoke_with_retries(
            self.llm, tmp_messages, max_tokens=30, temperature=0.2
        ).content

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

    # def gen_thoughts(self, outputs: Dict, messages: List[Dict]):
    #     return _gen_thoughts_for_rule_agents(outputs, messages, self.llm)

    def gen_explanation(self, outputs: Dict, messages: List[Dict]):
        """Generate explanation and update message list"""
        if self.optimize_thoughts_only:
            _gen_explanation(outputs, messages, self.llm, use_thoughts=self.use_thoughts)
        else:
            _gen_explanation_rules(outputs, messages, self.llm, use_thoughts=self.use_thoughts)


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
