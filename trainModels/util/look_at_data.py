import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pdb
from utilities_module import *
# import pdb

# Load data
data_path = '/groups/astuart/mtrautne/learnHomData/data/'
data_name = 'vor_res_data_256_tiny.pkl'
full_data = pkl.load(open(data_path + data_name, 'rb'))
A_input, chi1_true, chi2_true, x_ticks, y_ticks = full_data
print('A_input.shape = ', A_input.shape)
sgc = x_ticks.shape[1]
data_input, data_output = format_data(A_input,chi1_true, chi2_true,sgc)
# Plot a sample of each component of data_input and save it in Figures
samp_i = 0
fig, axes = plt.subplots(2,3, figsize=(20,10))
for i in range(3):
    axes[0,i].imshow(data_input[samp_i,i,:,:])
    axes[0,i].set_title('Component ' + str(i) + ' of data_input')
    axes[0,i].set_xticks([])
    axes[0,i].set_yticks([])

for j in range(2):
    axes[1,j].imshow(data_output[samp_i,j,:,:])
    axes[1,j].set_title('Component ' + str(j) + ' of data_output')
    axes[1,j].set_xticks([])
    axes[1,j].set_yticks([])

plt.savefig('../Figures/vor_data_input_output_res128.png')






