#!/bin/bash -l


#SBATCH --time=20:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH -J "smooth_data_1"    # job name
#SBATCH --output=LH_smooth_1.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu


pwd; hostname; date
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
module purge
module load matlab/r2020a

cd ../genData

matlab -r "data_solver('data_seeds/smooth_seeds.mat',@smooth_Amats,@smooth_fullfieldA,0.005,10000,128,'/groups/astuart/mtrautne/learnHomData/mat_data/smooth_data/')"

