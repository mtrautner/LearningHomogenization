#!/bin/bash


#SBATCH --time=00:45:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:0
#SBATCH --mem-per-cpu=64G
#SBATCH -J "pp_1"    # job name
#SBATCH --output=pp_%a.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array 0-4

cd ../trainModels/util
  
PARRAY=(32 64 256 512)
gridsize=${PARRAY[$SLURM_ARRAY_TASK_ID]}
echo $gridsize

python -u preprocess_mat_to_pkl.py "/groups/astuart/mtrautne/learnHomData/mat_data/smooth_res_data/smooth_$gridsize/" "/groups/astuart/mtrautne/learnHomData/data/" "smooth_res_data_${gridsize}" 500  
