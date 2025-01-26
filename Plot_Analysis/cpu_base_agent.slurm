#!/bin/bash -x

#SBATCH -n 8                       # Request 8 tasks
#SBATCH -N 1                        # Ensure the job runs on a single node
#SBATCH -p shared                   # Shared partition (adjust as appropriate for your cluster)
#SBATCH --mem=12G                   # Allocate 12 GB memory
#SBATCH -t 12:00:00                 # Job time limit
#SBATCH -o outputs/cpu_baseagent_%A_%a.out   # File to write STDOUT (%A inserts job ID, %a inserts array task ID)
#SBATCH -e outputs/cpu_baseagent_%A_%a.err   # File to write STDERR (%A inserts job ID, %a inserts array task ID)
#SBATCH --array=0-2                 # Array job with 3 tasks (index 0, 1, 2)

# Load Python module
module load python
source activate rulebots
module unload python

# Print current date and working directory for logging
date
pwd

# Define seeds array
SEEDS=(42 123 456)

# Get the seed for this array task
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Run the Python program with the corresponding seed and agent
python eval_llm.py --wandb-entity rulebots --track --seed $SEED --agent base_agent --exp_name v2

echo "Task $SLURM_ARRAY_TASK_ID with seed $SEED and agent 'base_agent' completed."




























# #SBATCH -n 8                       # Request 16 tasks
# #SBATCH -N 1                        # Ensure the job runs on a single node
# #SBATCH -p shared                   # Shared partition (adjust as appropriate for your cluster)
# #SBATCH --mem=12G                   # Allocate 24 GB memory
# #SBATCH -t 12:00:00                 # Job time limit
# #SBATCH -o outputs/cpu_runall_%A_%a.out   # File to write STDOUT (%A inserts job ID, %a inserts array task ID)
# #SBATCH -e outputs/cpu_runall_%A_%a.err   # File to write STDERR (%A inserts job ID, %a inserts array task ID)
# #SBATCH --array=0-2                 # Array job with 3 tasks (index 0, 1, 2)

# # Load Python module
# module load python
# source activate rulebots
# module unload python

# # Print current date and working directory for logging
# date
# pwd

# # Define seeds array
# SEEDS=(42 123 456)

# # Get the seed for this array task
# SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# # Run the Python program with the corresponding seed
# python sac_attention.py --wandb-entity rulebots --track --seed $SEED

# echo "Task $SLURM_ARRAY_TASK_ID with seed $SEED completed."