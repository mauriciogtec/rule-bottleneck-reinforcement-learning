import json
import re
from typing import Optional

from langchain_core.language_models import BaseChatModel


def gen_rules(
    llm: BaseChatModel,
    state_text: str,
    action_space_text: str,
    task_text: str,
    num_rules: int = 5,
    examples: Optional[str] = None,
    max_parse_attempts: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Generate a list of rules based on the environment.

    Args:
        llm (BaseChatModel): The language model to generate the rules.
        state_text (str): The current state of the environment.
        action_space_text (str): The possible actions in the environment.
        task_text (str): The task description.
        num_rules (int, optional): The number of rules to generate. Defaults to 5.
        examples (str, optional): Examples of rules. Defaults to None.
        max_parse_attempts (int, optional): The maximum number of attempts to parse the JSON. Defaults to 3.
        verbose (bool, optional): Whether to print the prompts and responses. Defaults to False.

    Returns:
        dict: A list of rules in a machine-readable format.
    """
    system_prompt = (
        "Your goal is to generate a set of *rules* that are useful to solve the resource-constrained allocation task "
        "given the current state of the decision problem."
        "Let's tackle this task step by step. "
        f"\n\n### Task:\n\n {task_text}"
        f"\n\n### Current state of the environment:\n\n {state_text}"
        f"\n\n### Possible actions:\n\n {action_space_text}"
    )
    if examples:
        system_prompt += (
            "\n\nBelow are some examples of rules that could be useful to solve the task. "
            f"\n\n### Examples\n\n{examples}\n\n"
        )

    thought_prompt = (
        "First, reason about what elements should be considered when choosing the optimal action "
        "the given task considering the task goal and optimal decision making. "
        "Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks "
        "of each action in the current state. Conclude the paragraph with a reflection of how they inform the design "
        "of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
        # "Your response should consist of two paragraphs. First a reflection of the possible consequences "
        # "of each action, and second, a reflection of the goals of the agents and how each of the given "
        # "priorization rules would apply to the given scenario."
    )

        # "Your response should consist of two paragraphs. First a reflection of the possible consequences "
        # "of each action, and second, a reflection of the goals of the agents and how certain rules "
        # "related to the task and goals would apply to the given scenario."
    )
    # send first call using the OpenAI API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": thought_prompt},
    ]

    thought_response = llm.invoke(messages).content

    if verbose:
        for m in messages:
            print(f"{m['role']}: {m['content']}")
        print("\n\nThoughts:\n")
        print(thought_response)

    rules_prompt = (
        f"Now, suggest {num_rules} rules that could be useful to solve the task. "
        " For each rule, provide the explanation of why it is important to consider it at the given state. "
        "Your response consist solely of a machine-readable JSON code."
        " This JSON structure should be a list with the follwing requirements: \n"
        "- Start with the character '[' and end with the character ']' \n"
        "- Each list entry should be a dictionary with the keys 'rule' and 'explanation'."
        "- The rules should be in natural language. While there is no strict format, it is recommended "
        " that they have the form 'If [condition], then [action], because [short justification]'."
        "- The explanation should expand on the rule justification and explain further how does it "
        "relate to the task and the goals of the decision maker, and what is the expected outcome of following the rule."
    )

    # send second call using the OpenAI API
    messages.extend(
        [
            {"role": "assistant", "content": thought_response},
            {"role": "user", "content": rules_prompt},
        ]
    )
    rules_response = llm.invoke(messages).content
    rules_response = _fix_common_json_list_errors(rules_response)

    if verbose:
        for m in messages:
            print(f"{m['role']}: {m['content']}")
        print("\n\nRules:\n")
        print(rules_response)

    try:
        # parse JSON
        rules = json.loads(rules_response)

    except Exception as e:
        # Try to fix the error
        attempts = 0
        error_message = str(e)
        while attempts < max_parse_attempts:
            try:
                # call OpenAI API
                fix_prompt = (
                    "You are a helpful assistant. Your task is to help fix the "
                    "syntax of machine-readable JSON file. You will be provided with an "
                    "error message that describes the issue when reading the JSON file. "
                    "Your response should be a corrected version of the JSON file without any "
                    "additional explanation so it can be parsed correctly."
                    f"\n\n### Error Message\n\n{error_message}"
                    f"\n\n### JSON File\n\n{rules_response}"
                )
                fix_messages = [
                    {"role": "system", "content": fix_prompt},
                ]
                rules_response = llm.invoke(fix_messages).content
                rules_response = _fix_common_json_list_errors(rules_response)

                rules = json.loads(rules_response)

                break
            except Exception as e:
                # increment attempts
                attempts += 1

                # update error message
                error_message = str(e)

        if attempts >= max_parse_attempts:
            raise ValueError(f"Failed to parse JSON: {error_message}")

    return rules


def call_for_action(
    llm: BaseChatModel,
    state_text: str,
    rules_text: list[dict],
    action_space_text: str,
    task_text: str,
    max_parse_attempts: int = 3,
    verbose: bool = False,
) -> tuple[int, str]:
    """
    Generate a call for action based on the environment.

    Args:
        llm (BaseChatModel): The language model to generate the rules.
        state_text (str): The current state of the environment.
        rules_text (str): The rules to consider.
        action_space_text (str): The possible actions in the environment.
        task_text (str): The task description.
        examples (str, optional): Examples of rules. Defaults to None.
        max_parse_attempts (int, optional): The maximum number of attempts to parse the JSON. Defaults to 3.
        verbose (bool, optional): Whether to print the prompts and responses. Defaults to False.

    Returns:
        int: The action to take.
        str: The explanation of the action.
    """

    # system prompt is same as the gen rules prompt, but instead of asking for rules
    # it focus on the optimal action only considering the priorization rules and their explanations

    system_prompt = (
        "Your goal is to choose the optimal action given the current state of the decision problem "
        "and the set of priorization rules that were generated to solve the resource-constrained allocation task. "
        "If no rule applies to the current state, you should consider the optimal rule without any priorization rule. "
        "Let's tackle this task step by step. "
        f"\n\n### Task:\n\n {task_text}"
        f"\n\n### Current state of the environment:\n\n {state_text}"
        f"\n\n### Possible actions:\n\n {action_space_text}"
        f"\n\n### Priorization rules:\n\n {rules_text}"
    )

    thought_prompt = (
        "First, reason about what elements should be considered when choosing the optimal action "
        "the given task considering the task goal and optimal decision making. "
        "Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks "
        "of each action in the current state. Conclude the paragraph with a reflection of how they inform the design "
        "of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
        # "Your response should consist of two paragraphs. First a reflection of the possible consequences "
        # "of each action, and second, a reflection of the goals of the agents and how each of the given "
        # "priorization rules would apply to the given scenario."
    )

    # send first call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": thought_prompt},
    ]

    thought_response = llm.invoke(messages).content

    if verbose:
        for m in messages:
            print(f"{m['role']}: {m['content']}")
        print("\n\nThoughts:\n")
        print(thought_response)

    action_prompt = (
        "Now, choose the optimal action given the current state of the environment and the set of priorization rules. "
        "Your response should consist of a single integer that corresponds to the index of the optimal action in the given list."
        "For example, the answer should be one of 0, 1, etc. with no additional explanation."
    )

    # send second call
    messages.extend(
        [
            {"role": "assistant", "content": thought_response},
            {"role": "user", "content": action_prompt},
        ]
    )

    action_response = llm.invoke(messages).content
    action = int(action_response)

    if verbose:
        for m in messages:
            print(f"{m['role']}: {m['content']}")
        print("\n\nAction:\n")
        print(action_response)

    explanation_prompt = (
        "Explain why you chose the optimal action given the current state of the environment and the set of priorization rules. "
        "Your response should be a short paragraph that explains the reasoning behind your choice."
    )

    # send third call
    messages.extend(
        [
            {"role": "user", "content": explanation_prompt},
        ]
    )

    explanation_response = llm.invoke(messages).content

    if verbose:
        for m in messages:
            print(f"{m['role']}: {m['content']}")
        print("\n\nExplanation:\n")
        print(explanation_response)

    return action, explanation_response


def _fix_common_json_list_errors(json_str: str) -> str:
    # 1. Remove the following patters
    patterns = ["```", "```json", "```yaml", "```yml", "\n"]
    json_str = re.sub("|".join(patterns), "", json_str)

    # 2. Remove trailing white space
    json_str = json_str.strip()

    # 3. Since the JSON is a list, make sure to being with '[' and end with ']'
    if not json_str.startswith("["):
        json_str = "[" + json_str
    if not json_str.endswith("]"):
        json_str = json_str + "]"

    # 4. Remove any white space after the '[', and ',', and before the ']'
    json_str = re.sub(r"\[\s+", "[", json_str)
    json_str = re.sub(r",\s+", ", ", json_str)
    json_str = re.sub(r"\s+\]", "]", json_str)

    return json_str


if __name__ == "__main__":
    import sys
    from langchain_together import ChatTogether, TogetherEmbeddings
    from weather2alert.env import HeatAlertEnv

    from src.language_wrappers import HeatAlertsWrapper

    # loead language based environment
    embed_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    env = HeatAlertEnv()
    env = HeatAlertsWrapper(env, embed_model)

    # reset environment
    obs, info = env.reset()
    state_text = info["obs_text"]

    # load LLM model
    llm_model = ChatTogether(model="meta-llama/Llama-3.2-3B-Instruct-Turbo")

    # obtain rules
    rules = gen_rules(
        llm_model, state_text, env.action_space_text, env.task_text, verbose=True
    )

    rules_text = str(rules)

    # obtain action
    action, explanation = call_for_action(
        llm_model,
        state_text,
        rules_text,
        env.action_space_text,
        env.task_text,
        verbose=True,
    )

    # print initial state and rules
    sys.exit(0)