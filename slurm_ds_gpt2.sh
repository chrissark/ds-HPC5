#!/bin/sh
#SBATCH -D /s/ls4/users/kristina/nlp/GPT-2
#SBATCH -o ./outputs/%j_ds_gpt2_nodes_5_gpus_4.out
#SBATCH -e ./outputs/%j_ds_gpt2_nodes_5_gpus_4.err
#SBATCH -p hpc5-gpu-3d
#SBATCH --nodes 5
#SBATCH --gres=gpu:k80:4
#SBATCH --ntasks-per-node 4

module load cuda/10.1

export HOME=/s/ls4/users/kristina
source $HOME/.bashrc
conda activate ds_env

mpirun python3 ds_gpt2.py
