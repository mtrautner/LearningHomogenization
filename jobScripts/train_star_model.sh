#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH -J "train_star_%a"    # job name
#SBATCH --output=train_star_%a.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=1

cd ../trainModels

python  -u train_model.py ./configs/star_config.yml $SLURM_ARRAY_TASK_ID  
