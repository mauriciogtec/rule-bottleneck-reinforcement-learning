import json
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
from langchain_core.language_models import BaseChatModel
from torch import nn

from layers import AttentionActor
from envs.language_wrappers import LanguageWrapper


class BaseAgent:
    """base class for the CoT agent with explanation"""

    def __init__(self, env: LanguageWrapper, llm: BaseChatModel):
        self.env = env
        self.llm = llm

    def system_prompt(self) -> str:
        task_text = self.env.task_text
        action_space_text = self.env.action_space_text

        return (
            f"\n\n### Task\n\n {task_text}"
            f"\n\n### Possible actions\n\n {action_space_text}"
        )

    def system_prompt_with_state(self, state: str, info: dict) -> str:
        system_prompt = self.system_prompt()
        system_prompt += f"\n\n### Current state of the environment\n\n {state}"
        return system_prompt

    def gen_thoughts(
        self, state: str, info: dict, prev_messages: List[Dict]
    ) -> dict:
        """Generate thoughts and update message list"""
        thought_prompt = (
            "First, reason about what elements should be considered when choosing the optimal action "
            "the given task considering the task goal and optimal decision making. "
            "Your response should consist of a single paragraph that reflects on the consequences, benefits, and drawbacks "
            "of each action in the current state. Conclude the paragraph with a reflection of how they inform the design "
            "of the priorization rules, and the different types of priorization rules that could be applied to the given scenario."
        )

        messages = prev_messages + [{"role": "system", "content": thought_prompt}]

        response = self.llm.invoke(messages, temperature=0.5, max_tokens=200).content

        messages += [{"role": "user", "content": response}]

        return {"thoughts": response, "messages": messages}

    def gen_explanation(
        self, state: str, info: dict, prev_messages: list
    ) -> Tuple[str, List[Dict]]:
        """Generate explanation and update message list"""
        explanation_prompt = (
            "Explain why you chose the optimal action given the current state of the environment and the set of priorization rules. "
            "Your response should be a short paragraph with 1-3 sentences that explain the reasoning behind your choice."
        )

        messages = prev_messages + [{"role": "system", "content": explanation_prompt}]

        response = self.llm.invoke(messages, temperature=0.5, max_tokens=200).content

        messages += [{"role": "user", "content": response}]

        return response, messages

    def gen_action(self, state: str, info: dict) -> Tuple[int, str]:
        """Generate a call for action based on the environment.

        Args:
            state_text (str): The current state of the environment.

        Returns:
            int: The action to take.
        """

        # make system prompt with the environment state
        system_prompt = self.system_prompt_with_state(state, info)
        messages = [{"role": "system", "content": system_prompt}]

        # get thoughts and update messages
        _, messages = self.gen_thoughts(state, info, messages)

        # get actions
        action_prompt = (
            "Now, choose the optimal action given the current state of the environment and the set of priorization rules. "
            "Your response should consist of a single integer that corresponds to the index of the optimal action in the given list."
            "For example, the answer should be one of 0, 1, etc. with no additional explanation."
            f"\n\n### Possible actions:\n\n {self.env.action_space_text}"
        )
        messages += [{"role": "user", "content": action_prompt}]
        response = self.llm.invoke(messages, max_tokens=20).content

        # use regex to extract the first integer
        action = re.search(r"\d+", response).group(0)
        action = int(action)

        # add explanation
        explanation, messages = self.gen_explanation(state, info, messages)

        return action, explanation, messages


class BaseRulesAgent(BaseAgent):
    """The rule-based agent generates a set of rules in addition to thoughts"""

    def __init__(
        self,
        env: LanguageWrapper,
        llm: BaseChatModel,
        num_rules: int = 5,
        example_rules: Optional[str] = None,
        max_parse_attempts: int = 3,
        verbose: bool = False,
    ):
        super().__init__(env, llm)
        self.num_rules = num_rules
        self.example_rules = example_rules
        self.max_parse_attempts = max_parse_attempts
        self.verbose = verbose

    def gen_rules(self, state: str, info: dict, messages: list[dict]) -> dict:
        """The rule-based agent generates a set of rules based on the environment state.

        Args:
            messages (list[dict]): A list of messages exchanged between the agent and the user.
                It is expected to the output of the `thoughts` method.
        """
        rules_prompt = (
            f"Now, suggest {self.num_rules} rules that could be useful to solve the task. "
            " For each rule, provide the explanation of why it is important to consider it at the given state. "
            "Your response consist solely of a machine-readable JSON code."
            " This JSON structure should be a list with the follwing requirements: \n"
            "- Start with the character '[' and end with the character ']' \n"
            "- Each list entry should be a dictionary with the keys 'rule' and 'explanation'."
            "- The rules should be in natural language. While there is no strict format, it is recommended "
            " that they have the form 'Prioritize [what] [when] [because (short justification)]'."
            "- The explanation should expand on the rule justification and explain further how does it "
            "relate to the task and the goals of the decision maker, and what is the expected outcome of following the rule."
        )

        # send second call using the OpenAI API
        messages = messages + [{"role": "system", "content": rules_prompt}]

        rules_response = self.llm.invoke(messages).content
        rules_response = self._fix_common_json_list_errors(rules_response)

        if self.verbose:
            for m in messages:
                print(f"{m['role']}: {m['content']}")
            print("\n\nRules:\n")
            print(rules_response)

        try:
            # parse JSON
            rules = json.loads(rules_response)
            self._verify_rules(rules)

        except Exception as e:
            # Try to fix the error
            attempts = 0
            error_message = str(e)
            while attempts < self.max_parse_attempts:
                try:
                    # call OpenAI API
                    fix_prompt = (
                        "You are a helpful assistant. Your task is to help fix the "
                        "syntax of machine-readable JSON file. You will be provided with an "
                        "error message that describes the issue when reading the JSON file. "
                        "Your response should be a corrected version of the JSON file without any "
                        "additional explanation so it can be parsed correctly."
                        "The keys must be 'rule' and 'explanation'. If a key is missing, "
                        "please add it with an empty string value."  # TODO
                        f"\n\n### Error Message\n\n{error_message}"
                        f"\n\n### JSON File\n\n{rules_response}"
                    )
                    fix_messages = [
                        {"role": "system", "content": fix_prompt},
                    ]
                    rules_response = self.llm.invoke(fix_messages).content
                    rules_response = self._fix_common_json_list_errors(rules_response)

                    rules = json.loads(rules_response)
                    self._verify_rules(rules)

                    break
                except Exception as e:
                    # increment attempts
                    attempts += 1

                    # update error message
                    error_message = str(e)

            if attempts >= self.max_parse_attempts:
                raise ValueError(f"Failed to parse JSON: {error_message}")

        # # expand number of rules by making combinations
        # rules = self._generate_rule_combinations(rules)

        return rules

    def gen_thoughts(
        self, state: str, info: dict, prev_messages: List[Dict]
    ) -> Tuple[str, List[Dict], Dict]:
        """This wrapper generates the initial thoughts and then adds priorization rules."""
        thoughts, messages = super().gen_thoughts(state, info, prev_messages)

        # generate rules
        rules = self.gen_rules(state, info, messages)

        # append priorization rules
        thoughts += f"\n\n### Priorization Rules\n\n {rules}"
        messages[-1]["content"] = thoughts

        return thoughts, messages

    @staticmethod
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

    @staticmethod
    def _verify_rules(rules: list[dict]) -> None:
        if not isinstance(rules, list):
            raise ValueError("Rules must be a list of dictionaries.")
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError("Each rule must be a dictionary.")
            if "rule" not in rule:
                raise ValueError("Each rule must have a 'rule' key.")
            if "explanation" not in rule:
                raise ValueError("Each rule must have an 'explanation' key")


class RulesSelectorRLAgent(BaseRulesAgent):
    """The rule-based agent generates a set of rules based on the environment state."""

    def __init__(
        self,
        actor: AttentionActor,
        env: LanguageWrapper,
        llm: BaseChatModel,
        max_rule_combinations: int = 3,
        num_rules: int = 5,
        example_rules: Optional[str] = None,
        max_parse_attempts: int = 3,
        verbose: bool = False,
    ):
        super().__init__(
            env, llm, num_rules, example_rules, max_parse_attempts, verbose,
        )
        self.actor = actor
        self.max_rule_combinations = max_rule_combinations

    def _generate_embeddings_for_rules(
        self, rules: List[Tuple[Dict]]
    ) -> List[List[float]]:
        """
        Generate embeddings for each rule combination using a language embedding model.

        Args:
            rule_combinations (List[Dict]): List of all possible rules. Each rule is a dictionary with keys
                'rule' and 'explanation'.
            embeddings_model (Embeddings): The language model used for embeddings.

        Returns:
            Dict[Tuple[str, ...], np.ndarray]: A dictionary mapping rule combinations to embeddings.
        """
        embeddings = {}
        documents = [
            "Rules:\n" + combo["rule"] + "\n\nExplanations:\n" + combo["explanation"]
            for combo in rules
        ]
        embeddings = self.embedder.embed_documents(documents)
        return embeddings

    def _generate_rule_combinations(self, rules: List[Dict]) -> List[Dict]:
        """
        Generate all non-empty combinations of rules.

        Args:
            rules (List[Dict]): A list of K rules, each consist of a dictionry with keys
                'rule' and 'explanation'.

        Returns:
            List[Dict] list of all non-empty combinations of rules. Rules are appended via text as well as the explanation.
        """
        all_combinations = []
        for r in range(
            self.max_rule_combinations or len(rules)
        ):  # r: size of the combination (1 to K)
            for combs in combinations(rules, r + 1):
                combs_rules = "- " + "\n- ".join({x["rule"] for x in combs})
                combs_expl = "- " + "\n- ".join({x["explanation"] for x in combs})
                all_combinations.append(
                    {"rule": combs_rules, "explanation": combs_expl}
                )

        return all_combinations

    def gen_rules(self, state: str, info: dict, messages: list[dict]) -> dict:
        """Wrapper for generating rules that includes combinations of and embeddings for rules.
        First, generates combinations of rules, next it filters using an RL selector
        """
        # get all possible rules with combinations
        rules = super().gen_rules(state, info, messages)
        rules_with_combs = self._generate_rule_combinations(rules)

        # get the rules and state embeddings
        rule_emb = self._generate_embeddings_for_rules(rules_with_combs)
        state_emb = self.embedder.embed_query(state)

        # convert to torch tensors
        device = next(self.actor.parameters()).device
        rule_emb = torch.tensor(rule_emb, dtype=torch.float32).to(device)
        state_emb = torch.tensor(state_emb, dtype=torch.float32).to(device)

        # get the
        with torch.no_grad():
            # query (1, emb_dim,) keys (num_rules, emb_dim)
            query, keys = state_emb.unsqueeze(0), rule_emb

            # logits (1, num_rules) -> (num_rules, )
            sel_idx, _ = self.actor.sample(query, keys)

        # get the selected rule
        selected_rule = rules_with_combs[sel_idx.item()]

        return selected_rule


if __name__ == "__main__":
    import sys

    from langchain_together import ChatTogether, TogetherEmbeddings
    from weather2alert.env import HeatAlertEnv
    from layers import AttentionActor

    from envs.language_wrappers import HeatAlertsWrapper

    # loead language based environment
    embed_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    env = HeatAlertEnv()
    env = HeatAlertsWrapper(env, embed_model)

    # reset environment
    obs, info = env.reset()
    state_text = info["obs_text"]

    # load LLM model
    llm_model = ChatTogether(model="meta-llama/Llama-3.2-3B-Instruct-Turbo")

    # # obtain rules
    # rules = gen_rules(
    #     llm_model, state_text, env.action_space_text, env.task_text, verbose=True
    # )

    # rules_text = str(rules)

    # # obtain action
    # action, explanation = call_for_action(
    #     llm_model,
    #     state_text,
    #     rules_text,
    #     env.action_space_text,
    #     env.task_text,
    #     verbose=True,
    # )

    # # test LLM RL Agent
    # agent = RuleBasedLLM(env, llm_model)

    # # test call for action
    # action, explanation, messages = agent.call_for_action(state_text, info)
    # print(messages)

    # test RL agent
    actor = AttentionActor(
        state_dim=768,
        rule_dim=768,
        hidden_dim=32,
    )
    agent = RulesSelectorRLAgent(actor, env, llm_model, embed_model, 768)

    # test call for action
    action, explanation, messages = agent.gen_action(state_text, info)
    print("Action:", action)
    print("Explanation:", explanation)
    print("Messages:")
    print(messages)

    # print initial state and rules
    sys.exit(0)
