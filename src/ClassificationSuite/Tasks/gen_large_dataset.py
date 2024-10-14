import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='glotzer_pf')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    if args.dataset == 'glotzer_pf':

        # Set bounds and resolution for dataset. Set to 10,000 points with the
        # bounds used in the paper.
        xmin = 0.0
        xmax = 500.0
        xnum = 317
        ymin = 0.0
        ymax = 1.0
        ynum = 317

        # Create grid with desired resolution. 
        xs = np.linspace(xmin, xmax, xnum)
        ys = np.linspace(ymin, ymax, ynum)
        coords = np.array([(x, y) for x in xs for y in ys])
        labels = np.zeros((xnum * ynum, 1))
        data = np.hstack((coords, labels))

        # Assign labels, assuming PF = 0.6.
        constant = 3.0 * math.pi * math.pi * 4.05 / (0.6 * 4.0)
        for row in range(data.shape[0]):
            if data[row, 0] > 0 and data[row, 1] > constant / data[row, 0]:
                data[row, 2] = 1
            else:
                data[row, 2] = -1
        labels = data[:,2].reshape(-1,1)

        # Visualize dataset, if specified.
        if args.visualize:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(7,7))
            plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=plt.get_cmap('bwr'), s=2.0)
            plt.xlabel(r'$Pe$')
            plt.ylabel(r'$x_{A}$')
            plt.xlim(xmin=xmin, xmax=xmax)
            plt.ylim(ymin=ymin, ymax=ymax)
            plt.show()

        # Report shape of dataset.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/Modified/glotzer_pf_large.npy', data)