import matplotlib.pyplot as plt
import numpy as np

from ClassificationSuite.Models import *

def get_model(name, task):
    '''
        Helper method for instantiating a specified model. 
    '''
    if name == 'gpc' and 'morgan' not in task:
        return GPC(ard=False)
    if name == 'gpc_ard' and 'morgan' not in task:
        return GPC(ard=True)
    if name == 'gpr' and 'morgan' not in task:
        return GPR(ard=False)
    if name == 'gpr_ard' and 'morgan' not in task:
        return GPR(ard=True)
    if 'gpc' in name and 'morgan' in task:
        return TanimotoGPC()
    if 'gpr' in name and 'morgan' in task:
        return TanimotoGPR()
    if name == 'bkde':
        return BKDE()
    if name == 'knn':
        return KNN()
    if name == 'lp':
        return LP()
    if name == 'nn':
        return NN()
    if name == 'rf':
        return RF()
    if name == 'sv':
        return SV()
    if name == 'xgb':
        return XGB()
    
def visualize_model_output(dataset, chosen_points, y_pred, y_acq, size=0):

    # Set plotting parameters.
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    # Gather data for plotting.
    fig, axs = plt.subplots(1, 3, figsize=(13,5))
    x_min = np.min(dataset[:,0])
    x_max = np.max(dataset[:,0])
    y_min = np.min(dataset[:,1])
    y_max = np.max(dataset[:,1])

    # Plot 1: Show currently chosen points relative to ground truth data.
    axs[0].scatter(dataset[:,0], dataset[:,1], s=2.0, c=dataset[:,-1], cmap=plt.get_cmap('bwr'))
    if size > 0:
        axs[0].scatter(chosen_points[:,0], chosen_points[:,1], color='honeydew', edgecolor='black', linewidth=0.5)
        axs[0].scatter(chosen_points[-size:,0], chosen_points[-size:,1], color='springgreen', edgecolor='black', linewidth=0.5)
    else:
        axs[0].scatter(chosen_points[:,0], chosen_points[:,1], color='springgreen', edgecolor='black', linewidth=0.5)
    axs[0].set_xlabel(r'x$_{1}$')
    axs[0].set_ylabel(r'x$_{2}$')
    axs[0].set_xlim(xmin=x_min, xmax=x_max)
    axs[0].set_ylim(ymin=y_min, ymax=y_max)

    # Plot 2: Show currently chosen points relative to current model predictions.
    axs[1].scatter(dataset[:,0], dataset[:,1], s=2.0, c=y_pred, cmap=plt.get_cmap('bwr'))
    if size > 0:
        axs[1].scatter(chosen_points[:,0], chosen_points[:,1], color='honeydew', edgecolor='black', linewidth=0.5)
        axs[1].scatter(chosen_points[-size:,0], chosen_points[-size:,1], color='springgreen', edgecolor='black', linewidth=0.5)
    else:
        axs[1].scatter(chosen_points[:,0], chosen_points[:,1], color='springgreen', edgecolor='black', linewidth=0.5)
    axs[1].set_xlabel(r'x$_{1}$')
    axs[1].set_xlim(xmin=x_min, xmax=x_max)
    axs[1].set_ylim(ymin=y_min, ymax=y_max)

    # Plot 3: Show currently chosen points relative to current acquisition function.
    axs[2].scatter(dataset[:,0], dataset[:,1], s=2.0, c=y_acq, cmap=plt.get_cmap('viridis'))
    if size > 0:
        axs[2].scatter(chosen_points[:,0], chosen_points[:,1], color='honeydew', edgecolor='black', linewidth=0.5)
        axs[2].scatter(chosen_points[-size:,0], chosen_points[-size:,1], color='springgreen', edgecolor='black', linewidth=0.5)
    else:
        axs[2].scatter(chosen_points[:,0], chosen_points[:,1], color='springgreen', edgecolor='black', linewidth=0.5)
    axs[2].set_xlabel(r'x$_{1}$')
    axs[2].set_xlim(xmin=x_min, xmax=x_max)
    axs[2].set_ylim(ymin=y_min, ymax=y_max)

    # Show figure.
    plt.show()

def save_frames(dataset, chosen_points, y_pred, y_acq, size=0, path='.'):
    '''
        Method for saving the frames produced by a round of active learning or 
        space-filling. Used primarily for compiling images for publication.
    '''

    # Set plotting parameters.
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    # Gather data for plotting.
    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    x_min = np.min(dataset[:,0])
    x_max = np.max(dataset[:,0])
    y_min = np.min(dataset[:,1])
    y_max = np.max(dataset[:,1])

    color1 = '#91dbff'
    color2 = '#fe7c7c'
    point_color = '#FFED7C'
    point_size = 100.0

    # Plot 1: Show currently chosen points relative to ground truth data.
    colors = [color1 if y_ == 1 else color2 for y_ in dataset[:,-1]]
    axs[0].scatter(dataset[:,0], dataset[:,1], s=2.0, c=colors)
    if size > 0:
        axs[0].scatter(chosen_points[:,0], chosen_points[:,1], s=point_size, color=point_color, edgecolor='black', linewidth=1.5)
        axs[0].scatter(chosen_points[-size:,0], chosen_points[-size:,1], s=point_size, color=point_color, edgecolor='black', linewidth=1.5)
    else:
        axs[0].scatter(chosen_points[:,0], chosen_points[:,1], color=point_color, s=point_size, edgecolor='black', linewidth=1.5)
    axs[0].set_xlim(xmin=x_min, xmax=x_max)
    axs[0].set_ylim(ymin=y_min, ymax=y_max)

    # Plot 2: Show currently chosen points relative to current model predictions.
    colors = [color1 if y_ == 1 else color2 for y_ in y_pred]
    axs[1].scatter(dataset[:,0], dataset[:,1], s=2.0, c=colors)
    if size > 0:
        axs[1].scatter(chosen_points[:,0], chosen_points[:,1], s=point_size, color=point_color, edgecolor='black', linewidth=1.5)
        axs[1].scatter(chosen_points[-size:,0], chosen_points[-size:,1], s=point_size, color=point_color, edgecolor='black', linewidth=1.5)
    else:
        axs[1].scatter(chosen_points[:,0], chosen_points[:,1], color=point_color, s=point_size, edgecolor='black', linewidth=1.5)
    axs[1].set_xlim(xmin=x_min, xmax=x_max)
    axs[1].set_ylim(ymin=y_min, ymax=y_max)

    # Plot 3: Show currently chosen points relative to current acquisition function.
    axs[2].scatter(dataset[:,0], dataset[:,1], s=2.0, c=y_acq, cmap=plt.get_cmap('viridis'))
    if size > 0:
        axs[2].scatter(chosen_points[:,0], chosen_points[:,1], s=point_size, color=point_color, edgecolor='black', linewidth=1.5)
        axs[2].scatter(chosen_points[-size:,0], chosen_points[-size:,1], s=point_size, color=point_color, edgecolor='black', linewidth=1.5)
    else:
        axs[2].scatter(chosen_points[:,0], chosen_points[:,1], color=point_color, s=point_size, edgecolor='black', linewidth=1.5)
    axs[2].set_xlim(xmin=x_min, xmax=x_max)
    axs[2].set_ylim(ymin=y_min, ymax=y_max)

    # Remove labels and ticks from each figure.
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    # Show figure.
    plt.tight_layout()
    if size == 0:
        plt.savefig(f'{path}/sf_{chosen_points.shape[0]}', dpi=1000)
    else:
        plt.savefig(f'{path}/al_{int(chosen_points.shape[0] / size) - 1}', dpi=1000)
    plt.show()
