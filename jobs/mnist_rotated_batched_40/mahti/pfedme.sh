#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=pfedme
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

module --force purge
module load pytorch
source /projappl/project_2009050/mytorch/bin/activate
pip uninstall -y fedai
pip install git+https://github.com/Ahmed-Khaled-Saleh/fedai.git
cd /projappl/project_2009050/fmtl-graph

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/mytorch/lib/python3.11/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"
ts=$(date +%Y%m%d_%H%M%S)
srun python main.py --config ./cfgs/mnist_rotated_batched_40/pfedme.yaml --env_file ./.env --timestamp ${ts}
