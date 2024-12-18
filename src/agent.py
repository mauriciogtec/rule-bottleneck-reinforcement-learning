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
) -> list[str]:
    """
    Generate a list of rules based on the environment.

    Args:
        env (LanguageWrapper): The environment to generate rules from.
        num_rules (int, optional): The number of rules to generate. Defaults to 10.

    Returns:
        list[str]: A list of generated rules.
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
        "First, reason about what elements should be considered when designing priorization rules "
        "the given task considering the task goal and optimal decision making. "
        "Your response should consist of two paragraphs. First a reflection of the possible consequences "
        "of each action, and second, a reflection of the goals of the agents and how certain rules "
        "related to the task and goals would apply to the given scenario."
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

    # print initial state and rules
    print("=== State ===")
    print(state_text)

    # print rules
    print("=== Rules ===")
    print(rules)
