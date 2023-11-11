#!/bin/bash


#SBATCH --time=00:25:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:0
#SBATCH --mem-per-cpu=32G
#SBATCH -J "pp_1"    # job name
#SBATCH --output=pp_1.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu


cd ../trainModels/util
  
python -u preprocess_mat_to_pkl.py '/groups/astuart/mtrautne/learnHomData/mat_data/smooth_data/' '/groups/astuart/mtrautne/learnHomData/data/' 'smooth_data' 10000  
