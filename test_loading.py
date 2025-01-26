import torch
# import matplotlib.pyplot as plt
import os
from layers import AttentionNetwork


# Path to the checkpoint file
checkpoint_path = "checkpoints/sac_attention_Uganda__sac_attention__gpt-4o-mini-huit__1__1737407614.state"

# Check if the file exists
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Inspect the keys to understand its structure
print("Checkpoint Keys:")
print(checkpoint.keys())
