#!/bin/bash
#SBATCH --job-name=distributed_wm_training
#SBATCH --partition=gpu-a100-80g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err


module load cuda
module load python
module load anaconda

conda activate
conda init bash

cd /scratch/work/sehadn1/mawm/mains/


ts=$(date +%Y%m%d_%H%M%S)
srun torchrun --standalone --nnodes=1 --nproc_per_node=2 train_wm.py --config ../cfgs/findgoal/mawm/main/mawm-seq-40.yaml --env_file ../.env --timestamp ${ts}
