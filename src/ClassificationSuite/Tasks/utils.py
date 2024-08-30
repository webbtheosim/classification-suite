import json
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(task):
    ''' 
        Loads the specified dataset from wherever the
        method is called.
    '''

    # Get the absolute location of the utils.py script.
    path = Path(__file__).parent

    # Load the dataset according to this path.
    dataset = np.load(f'{path}/Datasets/{task}.npy')

    return dataset

def task_config(task):
    '''
        Provides the initial sample size, batch size, and
        number of rounds for active learning tasks.
    '''

    # Read JSON config file.
    path = Path(__file__).parent
    f = open(f'{path}/config.json')
    config = json.load(f)

    return config[task]

def load_metafeatures():
    '''
        Loads the results of metafeature computations for the entire
        set of tasks. Metafeatures are filtered out if they are NaN.
    '''

    # Load metafeatures.
    path = Path(__file__).parent
    metafeatures = pickle.load(open(f'{path}/Metafeatures/metafeatures.pickle', 'rb'))

    # Filter out NaN metafeatures for each tasks.
    metafeatures_filtered = {}
    for task in list(metafeatures.keys()):
        metafeatures_filtered[task] = {}
        features = metafeatures[task]
        for feature, value in metafeatures[task].items():
            if not math.isnan(value):
                metafeatures_filtered[task][feature] = value

    return metafeatures_filtered

def visualize(dataset, points=None, color_with=None):
    '''
        Visualizes the dataset that has been passed to it.
        If the dimensions are greater than 2, than a PCA
        is performed to map the dataset to two dimensions.
    '''

    # Decompose dataset.
    X = dataset[:,0:-1]
    y = dataset[:,-1]
    if color_with is not None:
        y = color_with

    # Check if features need to be reduced.
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        scaler.fit(X)
        pca.fit(X)
        X = pca.transform(scaler.transform(X))
        if points is not None:
            points = pca.transform(scaler.transform(points))

    # Plot data points colored by label.
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.figure(figsize=(7,7))
    plt.scatter(X[:,0], X[:,1], s=5.0, c=y, cmap=plt.get_cmap('bwr'))
    if points is not None:
        plt.scatter(points[:,0], points[:,1], s=20.0, c='springgreen', edgecolors='black', linewidths=1.0)
    plt.xlabel(r'$\phi_{1}$')
    plt.ylabel(r'$\phi_{2}$')
    plt.xlim(xmin=np.min(X[:,0]), xmax=np.max(X[:,0]))
    plt.ylim(ymin=np.min(X[:,1]), ymax=np.max(X[:,1]))
    plt.show()

if __name__ == '__main__':

    # Test load_data.
    dataset = load_data(task='qm9_zpve')
    X = dataset[:,0:-1]
    y = dataset[:,-1]
    print(f'Shape of features: {X.shape}')
    print(f'Shape of labels: {y.shape}')

    # Test visualize.
    visualize(dataset)