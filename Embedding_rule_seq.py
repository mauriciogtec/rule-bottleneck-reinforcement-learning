from itertools import combinations
from typing import List, Tuple, Dict
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_together import TogetherEmbeddings

def generate_rule_combinations(rules: List[str]) -> List[Tuple[str, ...]]:
    """
    Generate all non-empty combinations of rules.

    Args:
        rules (List[str]): A list of K rules.

    Returns:
        List[Tuple[str, ...]]: A list of all non-empty combinations of rules.
    """
    all_combinations = []
    for r in range(1, len(rules) + 1):  # r: size of the combination (1 to K)
        all_combinations.extend(combinations(rules, r))
    return all_combinations


def generate_embeddings_for_rules(
    rule_combinations: List[Tuple[str, ...]], embeddings_model: Embeddings
) -> Dict[Tuple[str, ...], np.ndarray]:
    """
    Generate embeddings for each rule combination using a language embedding model.

    Args:
        rule_combinations (List[Tuple[str, ...]]): List of rule combinations.
        embeddings_model (Embeddings): The language model used for embeddings.

    Returns:
        Dict[Tuple[str, ...], np.ndarray]: A dictionary mapping rule combinations to embeddings.
    """
    embeddings = {}
    for combo in rule_combinations:
        # Combine the rules into a single string
        combined_text = " | ".join(combo)
        # Generate embedding for the combined text
        embedding = embeddings_model.embed_query(combined_text)
        # Convert to NumPy array for further processing
        embeddings[combo] = np.array(embedding, dtype=np.float32)
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
