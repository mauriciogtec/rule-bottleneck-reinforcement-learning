import tyro
import wandb
from dataclasses import dataclass

from typing import Any, Dict, List, Optional


class Args:
    """Arguments for the fidelity experiments."""

    # The name of the experiment
    wandb_run_path: str = "rulebots/rulebots-compare/0v3ueac5"
    # The number of samples to generate
    num_samples: int = 1000
    # The number of dimensions of the samples
    num_dimensions: int = 10
    # The number of clusters to generate
    num_clusters: int = 3
    # The random seed for reproducibility
    random_seed: int = 42
    # The output directory for saving results
    output_dir: str = "./results"
    # Whether to save the generated samples to a file
    save_samples: bool = True


def main(args: Args) -> None:
    """Main function to run the fidelity experiments."""
    # Set the random seed for reproducibility
    np.random.seed(args.random_seed)

    # Generate synthetic data
    data = generate_synthetic_data(
        num_samples=args.num_samples,
        num_dimensions=args.num_dimensions,
        num_clusters=args.num_clusters,
        random_seed=args.random_seed,
    )

    # Save the generated samples to a file if specified
    if args.save_samples:
        save_samples_to_file(data, args.output_dir)

    print(f"Experiment '{args.experiment_name}' completed successfully.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
