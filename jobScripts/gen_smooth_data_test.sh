#!/bin/bash -l


#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH -J "smooth_data_%a"    # job name
#SBATCH --output=LH_smooth_%a.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array 0-3

pwd; hostname; date
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
module purge
module load matlab/r2020a

PARRAY=(32 64 256 512)
gridsize=${PARRAY[$SLURM_ARRAY_TASK_ID]}

cd ../genData

matlab -r "data_solver('data_seeds/smooth_seeds_test_only.mat',@smooth_Amats,@smooth_fullfieldA,0.005,500,$gridsize,'/groups/astuart/mtrautne/learnHomData/mat_data/smooth_res_data/smooth_$gridsize/')"

