#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH -J "train_smooth_hyp_%a"    # job name
#SBATCH --output=hyperparam_output/train_smooth_%a.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=1

cd ../trainModels

python  -u train_model.py ./configs/hyperparam_configs/smooth/smooth_model_size_${SLURM_ARRAY_TASK_ID}.yaml  
