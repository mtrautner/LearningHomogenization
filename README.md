# LearningHomogenization
Repository associated to the paper "Learning Homogenization for Elliptic Operators"


# Requirements


# Generating Data

Smooth: in the genData directory in MATLAB, run 

>> data_solver('data_seeds/smooth_seeds.mat',@smooth_Amats,@smooth_fullfieldA,0.005,1,128,'data/smooth_data/')


to generate one data solution. The output may be visualized by using the plot_solutions function in /utils/. To generate all the data, run 

>> data_solver('data_seeds/smooth_seeds.mat',@smooth_Amats,@smooth_fullfieldA,0.005,10000,128,'data/smooth_data/')

Note: the data will take up several GB and take a while to generate. It is highly recommended to uncomment the parpool(6) line, ensure <for> is <parfor> and run on a cluster.

The equivalent data generation commands for the other microstructures are

Star: >> data_solver('data_seeds/star_inc_seeds.mat',@star_Amats,@star_fullfieldA,0.005,10000,128,'data/star_data/')

Square: >> data_solver('data_seeds/sq_inc_seeds.mat',@sq_Amats,@sq_fullfieldA,0.005,10000,128,'data/sq_data/')

Voronoi: >> data_solver('data_seeds/vor_seeds.mat',@vor_Amats,@vor_fullfieldA,0.005,10000,128,'data/vor_data/')

## Preprocess Data
The .mat files must be converted to .pkl files. To do so, run preprocess_mat_to_pkl.py in trainModels/util
with four arguments: 
arg1: directory of .mat data
arg2: directory to save .pkl data
arg3: name of .pkl file
arg4: N_data

This will also create a smaller .pkl file of just 20 data samples for easy debugging. 

# Train Model

