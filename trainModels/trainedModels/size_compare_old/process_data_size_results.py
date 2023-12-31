'''
This script takes the error files for 
smooth and voronoi microstructures for all 5 samples
with each data size and plots the mean and std of the errors
'''

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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

plt.rcParams['font.family'] = 'serif'  # or 'DejaVu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 'DejaVu Serif' 'serif' 'Times
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
'''

tickfontsize = 30
fontsize = 30
linewidth = 4
markersize = 15

SMALL_SIZE = tickfontsize
MEDIUM_SIZE = tickfontsize
BIGGER_SIZE = fontsize

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

shapes = ['o','s']

# Get the errors for each data size
data_sizes = [2000, 4000, 6000, 8000,9500]
smooth_errors = np.zeros((5, 5))
vor_errors = np.zeros((5, 5))

for sample in range(5):
    for datasize in data_sizes[:-1]:
        smooth_error_file = 'smooth_data_' + str(datasize) + '_' + str(sample) + '_errors.yml'
        # read error file and extract H1 mean error
        with open(smooth_error_file, 'r') as f:
            error_dict = yaml.load(f, Loader=yaml.FullLoader)
            smooth_errors[sample, data_sizes.index(datasize)] = error_dict['H1_mean']
        vor_error_file = 'vor_data_' + str(datasize) + '_' + str(sample) + '_errors.yml'
        # read error file and extract H1 mean error
        with open(vor_error_file, 'r') as f:
                error_dict = yaml.load(f, Loader=yaml.FullLoader)
                vor_errors[sample, data_sizes.index(datasize)] = error_dict['H1_mean']

# Add the errors for the 9500 data size
for sample in range(5):
    smooth_error_file = '../standard_models/smooth_model_' + str(sample) + '_errors.yml'
    # read error file and extract H1 mean error
    with open(smooth_error_file, 'r') as f:
        error_dict = yaml.load(f, Loader=yaml.FullLoader)
        smooth_errors[sample, 4] = error_dict['H1_mean']
    vor_error_file = '../standard_models/vor_model_' + str(sample) + '_errors.yml'
    # read error file and extract H1 mean error
    with open(vor_error_file, 'r') as f:
            error_dict = yaml.load(f, Loader=yaml.FullLoader)
            vor_errors[sample, 4] = error_dict['H1_mean']

# Get the mean and std of the errors
smooth_mean_errors = np.mean(smooth_errors, axis=0)
smooth_std_errors = np.std(smooth_errors, axis=0)
vor_mean_errors = np.mean(vor_errors, axis=0)
vor_std_errors = np.std(vor_errors, axis=0)

# Plot the results
plt.figure(figsize=(10,10))
plt.plot(data_sizes, vor_mean_errors, label='Voronoi',marker = shapes[1],color = CB_color_cycle[0],linewidth=linewidth,markersize=markersize)
plt.plot(data_sizes, smooth_mean_errors, label='Smooth',marker = shapes[0],color = CB_color_cycle[1],linewidth=linewidth,markersize=markersize)
plt.errorbar(data_sizes, vor_mean_errors, yerr=2*vor_std_errors,color = CB_color_cycle[0])
plt.errorbar(data_sizes, smooth_mean_errors, yerr=2*smooth_std_errors,color = CB_color_cycle[1])
plt.fill_between(data_sizes, vor_mean_errors - 2*vor_std_errors, vor_mean_errors + 2*vor_std_errors, alpha=0.2, color = CB_color_cycle[0])
plt.fill_between(data_sizes, smooth_mean_errors - 2*smooth_std_errors, smooth_mean_errors + 2*smooth_std_errors, alpha=0.2, color = CB_color_cycle[1])
plt.legend(fontsize=fontsize)
plt.xlabel('Number of Training Samples',fontsize=fontsize)
plt.ylabel(r'Mean $H^1$ Error',fontsize=fontsize)
# set xticks
plt.xticks(data_sizes,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
#log scales
plt.yscale('log')
plt.xscale('log')
# get log-log slope
x = np.log(data_sizes)
y = np.log(smooth_mean_errors)
m, b = np.polyfit(x, y, 1)
print('Smooth slope: ', m)
# add smoth slope to plot
plt.text(0.5,0.22,r'Smooth slope: ' + str(round(m,2)),fontsize=fontsize,transform=plt.gca().transAxes)
y = np.log(vor_mean_errors)
m, b = np.polyfit(x, y, 1)
print('Voronoi slope: ', m)
# add voronoi slope to plot
plt.text(0.18,0.82,r'Voronoi slope: ' + str(round(m,2)),fontsize=fontsize,transform=plt.gca().transAxes)
plt.title('QQQ', color = 'white',fontsize=fontsize)
# set xticks, not scientific notation
# remove extra xticks

# make not scientific notation
# Plot a line with slope -1/2
x = np.linspace(2000,9500,100)
y = 0.5*x**(-1/2)
plt.plot(x,y,linestyle='--',color='black',linewidth=linewidth)
plt.text(0.5,0.48,r'$\mathcal{O}(N^{-1/2})$',fontsize=fontsize,transform=plt.gca().transAxes)

plt.savefig('../../Figures/data_size_compare.pdf',bbox_inches='tight')
