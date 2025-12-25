#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=data_collection
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module --force purge
module load pytorch
source /scratch/project_2009050/torchy/bin/activate

cd /projappl/project_2009050/mawm/
cd ./mains

export PYTHONPATH=$PYTHONPATH:/scratch/project_2009050/torchy/lib/python3.12/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"

srun python collect_data.py