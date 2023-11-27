import yaml
import numpy as np
import matplotlib.pyplot as plt
import pdb
"""
Read error .yml files and process them into a table for comparison between different hyperparameters
"""

errors = np.zeros((144, 4))
b_sizes = np.zeros((144, 1))
modes = np.zeros((144, 1))
lrs = np.zeros((144, 1))
epochs = np.zeros((144, 1))

for index in range(144):
    index = index + 1
    try:
        config = yaml.load(open('vor_model_size_' + str(index) + '_config.yml', 'r'), Loader=yaml.FullLoader)
        b_sizes[index - 1] = config['batch_size']
        modes[index - 1] = config['N_modes']
        lrs[index - 1] = config['lr']
        epochs[index - 1] = config['epochs']
        error_dict = yaml.load(open('vor_model_size_' + str(index) + '_errors.yml', 'r'), Loader=yaml.FullLoader)
        errors[index - 1, 0] = error_dict['H1_mean']
        errors[index - 1, 1] = error_dict['H1_rel_mean']
        errors[index - 1, 2] = error_dict['W1_10_rel_med']
        errors[index - 1, 3] = error_dict['Abar_rel_error2_med']
    except:
        print('Error in reading file ' + str(index))
        errors[index - 1, :] = 0.1

err_ind =2
# Create error heatmaps for pairs of hyperparameters
# Batch size vs. modes
fig, ax = plt.subplots(3,2)
ax00 = ax[0,0].scatter(b_sizes, modes, c=errors[:,err_ind ])
ax[0,0].set_xlabel('Batch size')
ax[0,0].set_ylabel('Modes')
# add colorbar
fig.colorbar(ax00,ax = ax[0,0])

# Batch size vs. learning rate
ax01 = ax[0,1].scatter(b_sizes, lrs, c=errors[:, err_ind])
ax[0,1].set_xlabel('Batch size')
ax[0,1].set_ylabel('Learning rate')
# add colorbar 
fig.colorbar(ax01,ax = ax[0,1])

# Batch size vs. epochs
ax10 = ax[1,0].scatter(b_sizes, epochs, c=errors[:, err_ind])
ax[1,0].set_xlabel('Batch size')
ax[1,0].set_ylabel('Epochs')
# add colorbar
fig.colorbar(ax10,ax = ax[1,0])

# Epochs vs learning rate
ax11 = ax[1,1].scatter(epochs, lrs, c=errors[:, err_ind])
ax[1,1].set_xlabel('Epochs')
ax[1,1].set_ylabel('Learning rate')
# add colorbar
fig.colorbar(ax11,ax = ax[1,1])

# Modes vs learning rate
ax20 = ax[2,0].scatter(modes, lrs, c=errors[:, err_ind])
ax[2,0].set_xlabel('Modes')
ax[2,0].set_ylabel('Learning rate')
# add colorbar
fig.colorbar(ax20,ax = ax[2,0])

# Modes vs epochs
ax21 = ax[2,1].scatter(modes, epochs, c=errors[:, err_ind])
ax[2,1].set_xlabel('Modes')
ax[2,1].set_ylabel('Epochs')
# add colorbar
fig.colorbar(ax21,ax = ax[2,1])


# Label axes with modes corresponding to each row/column
# Add space between figures
fig.tight_layout(pad=1.0)
plt.savefig('errors_bs_modes.png')
