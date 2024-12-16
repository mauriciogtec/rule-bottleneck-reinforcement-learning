from langchain_together import Together
from langchain_core.language_models import LLM


def gen_rules(
    llm: LLM, state: str, task_description: str, num_rules: int = 10, examples: str | None = None
) -> list[str]:
    """
    Generate a list of rules based on the environment.

    Args:
        env (LanguageWrapper): The environment to generate rules from.
        num_rules (int, optional): The number of rules to generate. Defaults to 10.

    Returns:
        list[str]: A list of generated rules.
    """
    # task_description = env.task_description
    instruction = (
        "Your goal is to generate a set of *rules* that are useful to solve the resource-constrained allocation task "
        "given the current state of the decision problem."
        "Let's think step by step. "
        "First, reason about what elements should be considered when designing priorization rules "
        "the given task. Next, elaborate on what rules could apply to the given state to make an optimal decision. "
        f"Last, suggest {num_rules} rules that could be useful to solve the task. For each rule, provide "
        "the rationale of why it is important to consider it at the given state and what would be the goal of applying it "
        "to the decision problem and the expected outcome."
        f"\nThe final list of rules should be given in YAML format. The YAML dictionary must contain a list of {num_rules} entries "
        "where each entry is a dictionary with the following keys: 'rule', 'rationale'. "
        "\nThe YAML dictionary must be enclose with triple quotes ```yaml <list> ```"
    )

    prompt = (
        f"### Task Description\n\n{task_description}\n\n"
        f"### Instructions\n\n{instruction}\n\n"
    )

    if examples:
        prompt += f"### Examples\n\n{examples}\n\n"

    prompt += f"### Current state\n\n{state}\n\n"
    prompt += f"### Your response\n\n"

    response = llm.invoke(prompt)

    return response


if __name__ == "__main__":
    from src.language_wrappers import HeatAlertsWrapper
    from weather2alert.env import HeatAlertEnv
    from langchain_together import TogetherEmbeddings

    embed_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    env = HeatAlertEnv()
    env = HeatAlertsWrapper(env, embed_model)
    task_description = HeatAlertsWrapper.task_description
    obs, info = env.reset()
    state = info["obs_text"]

    llm_model = Together(model="meta-llama/Llama-3.2-3B-Instruct-Turbo")

    rules = gen_rules(llm_model, state, task_description)
