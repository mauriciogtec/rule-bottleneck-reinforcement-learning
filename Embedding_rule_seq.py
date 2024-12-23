from itertools import combinations
from typing import List, Optional, Tuple, Dict
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_together import TogetherEmbeddings


class Agent:
    def __init__(self, env, policy_network):
        self.env = env


def generate_rule_combinations(
    rules: List[Dict], max_combs: Optional[int] = None
) -> List[Tuple[str, ...]]:
    """
    Generate all non-empty combinations of rules.

    Args:
        rules (List[Dict]): A list of K rules, each consist of a dictionry with keys
            'rule' and 'explanation'.

    Returns:
        List[Dict] list of all non-empty combinations of rules. Rules are appended via text as well as the explanation.
    """
    all_combinations = []
    for r in range(max_combs or len(rules)):  # r: size of the combination (1 to K)
        combs = combinations(rules, r + 1)
        combs_rules = "- " + "\n - ".join({x["rule"] for x in combs})
        combs_expl = "- " + "\n - ".join({x["explanation"] for x in combs})
        all_combinations.extend({"rule": combs_rules, "explanation": combs_expl})
    return all_combinations


def generate_embeddings_for_rules(
    rule_combinations: List[Tuple[Dict]], embeddings_model: Embeddings
) -> Dict[Tuple[str, ...], np.ndarray]:
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
        "Rules:\n" + combo["rule"] + "\n\nExplanation:\n" + combo["explanation"]
        for combo in rule_combinations
    ]
    embeddings = embeddings_model.embed_documents(documents)
    return embeddings


if __name__ == "__main__":
    # Example list of rules
    rules = [
        "Prioritize the hot weather",
        "Prioritize the abnormal weather",
        "Alert based on regional temperature spikes",
        "Focus on high-risk areas",
    ]

    # Step 1: Generate all rule combinations
    rule_combinations = generate_rule_combinations(rules)
    print(f"Generated {len(rule_combinations)} rule combinations:")
    for combo in rule_combinations:
        print(combo)

    # Step 2: Load embedding model
    model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

    # Step 3: Generate embeddings for all rule combinations
    embeddings_dict = generate_embeddings_for_rules(rule_combinations, model)

    # Output embeddings
    for combo, embedding in embeddings_dict.items():
        print(f"Rule Combination: {combo}")
        print(f"Embedding Shape: {embedding.shape}")
        print(f"Embedding Vector: {embedding[:5]}...")  # Print the first 5 values
