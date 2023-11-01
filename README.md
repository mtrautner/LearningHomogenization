# LearningHomogenization
Repository associated to the paper "Learning Homogenization for Elliptic Operators"


# Requirements


# Generating Data

Smooth: in the genData directory in MATLAB, run 

>> data_solver('data_seeds/smooth_seeds.mat',@smooth_Amats,@smooth_fullfieldA,0.005,1,128,'data/smooth_data/',1)


to generate one data solution and visualize the input and output. To generate all the data (turning off visualization), run 

>> data_solver('data_seeds/smooth_seeds.mat',@smooth_Amats,@smooth_fullfieldA,0.005,10000,128,'data/smooth_data/',0)


The equivalent data generation commands for the other microstructures are

Star: >> data_solver('data_seeds/star_inc_seeds.mat',@star_Amats,@star_fullfieldA,0.005,10000,128,'data/star_data/',0)

Square: >> data_solver('data_seeds/sq_inc_seeds.mat',@sq_Amats,@sq_fullfieldA,0.005,10000,128,'data/sq_data/',0)

Voronoi: >> data_solver('data_seeds/vor_seeds.mat',@vor_Amats,@vor_fullfieldA,0.005,10000,128,'data/vor_data/',0)
