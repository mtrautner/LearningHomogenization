import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import yaml

# Set font default
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
matplotlib.rcParams['mathtext.rm'] = 'stix'
matplotlib.rcParams['mathtext.it'] = 'stix'
matplotlib.rcParams['mathtext.bf'] = 'stix'
'''
    This script is used to process the results of the size comparison experiment.
    Takes the error.yaml files from each model and plots the results.
'''
modes = [6,12,18,24,36,48]
model_sizes = [144,288,576,1152]

smooth_yaml_path = 'smooth_model_size_'
vor_yaml_path = 'vor_model_size_'
smooth_errors = np.zeros((24, 5))
smooth_errors_by_mode_and_size = np.zeros((6,4,5))
vor_errors = np.zeros((24, 5))
vor_errors_by_mode_and_size = np.ones((6,4,5))*0.05

for m_index in range(24):
    # Figure out the save index
    for sample in range(5):
        smooth_error_file = smooth_yaml_path + str(m_index) + '_' + str(sample) + '_errors.yml'
        # read error file and extract H1 mean error
        with open(smooth_error_file, 'r') as f:
            error_dict = yaml.load(f, Loader=yaml.FullLoader)
            smooth_errors[m_index, sample] = error_dict['H1_rel_mean']
        vor_error_file = vor_yaml_path + str(m_index) + '_' + str(sample) + '_errors.yml'
        # read error file and extract H1 mean error
        try:
            with open(vor_error_file, 'r') as f:
                error_dict = yaml.load(f, Loader=yaml.FullLoader)
                vor_errors[m_index, sample] = error_dict['H1_rel_mean']
        except:
            pass
        # Figure out the mode and size
        smooth_config_file = smooth_yaml_path + str(m_index) + '_' + str(sample) + '_config.yml'
        with open(smooth_config_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            mode = config_dict['N_modes']
            width = config_dict['width']
            size = mode * width
            mode_index = modes.index(mode)
            size_index = model_sizes.index(size)
            smooth_errors_by_mode_and_size[mode_index, size_index, sample] = smooth_errors[m_index, sample]
        vor_config_file = vor_yaml_path + str(m_index) + '_' + str(sample) + '_config.yml'
        try:
            with open(vor_config_file, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
                mode = config_dict['N_modes']
                width = config_dict['width']
                size = mode * width
                mode_index = modes.index(mode)
                size_index = model_sizes.index(size)
                vor_errors_by_mode_and_size[mode_index, size_index, sample] = vor_errors[m_index, sample]
        except:
            pass
# Get the mean and std of the errors
smooth_mean_errors = np.mean(smooth_errors_by_mode_and_size, axis=2)
smooth_std_errors = np.std(smooth_errors_by_mode_and_size, axis=2)
vor_mean_errors = np.mean(vor_errors_by_mode_and_size, axis=2)
vor_std_errors = np.std(vor_errors_by_mode_and_size, axis=2)
print(min(smooth_mean_errors.flatten()))
print(min(vor_mean_errors.flatten()))
# print m_index of min error
print(np.argmin(smooth_mean_errors.flatten()))
print(np.argmin(vor_mean_errors.flatten()))
print(vor_errors_by_mode_and_size)
# Plot the results

fig, axes = plt.subplots(1,2, figsize=(20,10))
# Make all font size bigger
fontsize = 30
y_max = max(max(smooth_mean_errors.flatten()), max(vor_mean_errors.flatten()))
y_min = min(min(smooth_mean_errors.flatten()), min(vor_mean_errors.flatten()))

for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    ax.set_xlabel('Modes', fontsize=fontsize)
    # set modes ticks at modes
    ax.set_xticks(modes)
    # logscale
    ax.set_yscale('log')
    # set y limits
    ax.set_ylim([y_min, y_max])
    ax.set_ylabel(r'Mean Relative $H^1$ Error', fontsize=fontsize)
    # legend upper left
# Put markers on the points same color as lines
colors = CB_color_cycle[0:4]
shapes = ['o', 's', 'D', 'v']
for i in range(4):
    ax1 = axes[0]
    ax1.plot(modes, smooth_mean_errors[:,i], marker=shapes[i], label='Size = ' + str(model_sizes[i]), color=colors[i], markersize=8, linewidth=2.5)
    ax1.errorbar(modes, smooth_mean_errors[:,i], yerr=smooth_std_errors[:,i], color=colors[i])
    # fill between for errors
    ax1.fill_between(modes, smooth_mean_errors[:,i] - smooth_std_errors[:,i], smooth_mean_errors[:,i] + smooth_std_errors[:,i], alpha=0.2, color=colors[i])
    ax1.set_title('Smooth Microstructure', fontsize=fontsize)
    ax2 = axes[1]
    ax2.plot(modes, vor_mean_errors[:,i], marker=shapes[i], label='Size = ' + str(model_sizes[i]), color=colors[i], markersize=8, linewidth=2.5)
    ax2.errorbar(modes, vor_mean_errors[:,i], yerr=vor_std_errors[:,i], color=colors[i])
    # fill between for errors
    ax2.fill_between(modes, vor_mean_errors[:,i] - vor_std_errors[:,i], vor_mean_errors[:,i] + vor_std_errors[:,i], alpha=0.2, color=colors[i])
    ax2.set_title('Voronoi Microstructure', fontsize=fontsize)
    
# Top Left legend
axes[0].legend(loc=2, fontsize=fontsize)
# Bottom Right legend
axes[1].legend(loc=3, fontsize=fontsize)

# Pad space between subplots
plt.subplots_adjust(wspace=0.4)

plt.savefig('../../Figures/size_compare.pdf')
plt.savefig('../../Figures/size_compare.svg')


