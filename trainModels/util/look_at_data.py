import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
# import pdb

data_path = '/groups/astuart/mtrautne/Singularities/'
with open(data_path + 'data/voronoi_data_54tiny.pkl', 'rb') as handle:
    A_input, chi1_true, chi2_true, x_ticks, y_ticks = pkl.load(handle)

sgc = x_ticks.shape[1] # Ex: 512
# Plot A, chi1, chi2 and save as pdf on a plot with 3 subplots
print(A_input.shape)
print(chi1_true.shape)
print(chi2_true.shape)
fig, axs = plt.subplots(1,3, figsize = (15,5))
axs[0].imshow(np.reshape(A_input[0,:,0,0],(sgc, sgc)))
axs[0].set_title('A')
axs[1].imshow(np.reshape(chi1_true[:,0],(sgc, sgc)))
axs[1].set_title('chi1')
axs[2].imshow(np.reshape(chi2_true[:,0],(sgc, sgc)))
axs[2].set_title('chi2')
plt.savefig('/home/mtrautne/Singularities/Figures/vor_54_data_sample.pdf')
