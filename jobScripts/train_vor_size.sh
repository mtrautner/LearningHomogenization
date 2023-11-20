#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH -J "train_vor_size_%a"    # job name
#SBATCH --output=size_output/train_vor_size_%a.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=0-119

cd ../trainModels

for ip1 in {0..4} # 5 samples
do 
  for i in {0..23} # 24 options
  do 
     let task_id=$i*5+$ip1
     printf $task_id"\n"
     if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
     then
	python  -u train_model.py ./configs/size_configs/vor/vor_model_size_${i}_${ip1}.yaml  
     fi
  done
done
