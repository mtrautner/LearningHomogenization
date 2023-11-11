#!/bin/bash -l


#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH -J "vor_data_4"    # job name
#SBATCH --output=LH_vor_%a.out  
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

matlab -r "data_solver('data_seeds/vor_seeds.mat',@vor_Amats,@vor_fullfieldA,0.005,500,$gridsize,'/groups/astuart/mtrautne/learnHomData/mat_data/vor_res_data/vor_$gridsize/')"

