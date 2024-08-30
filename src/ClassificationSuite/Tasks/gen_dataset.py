#====================================================
#   SCRIPT FOR DATASET CONSTRUCTION AND REPRODUCTION
#====================================================

import argparse
from bs4 import BeautifulSoup
import imageio.v3 as iio
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mordred import Calculator, descriptors
import numpy as np
import os
import pandas as pd
import pickle
import random
import rdkit.Chem as Chem
import requests
from sbs.densitysampling import DensitySampler as dsam
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from ternary_diagram import TernaryDiagram
import wget

if __name__ == '__main__':

    # Get user input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=[
            'princeton',
            'glotzer_pf',
            'glotzer_xa',
            'water_lp',
            'water_hp',
            'oxidation',
            'shower',
            'vdw',
            'diblock',
            'bear',
            'electro',
            'hplc',
            'oer',
            'toporg',
            'polygel',
            'polysol',
            'perovskite',
            'qm9_gap',
            'qm9_r2',
            'qm9_cv',
            'qm9_zpve',
            'qm9_u0',
            'robeson',
            'free',
            'esol',
            'lipo',
            'hiv',
            'bace',
            'clintox',
            'muv',
            'tox21'
        ],
        default='princeton'
    )
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # Generate the Princeton P dataset.
    if args.dataset == 'princeton':

        # Load relevant PNG file and process.
        wget.download('https://upload.wikimedia.org/wikipedia/commons/1/15/Princeton_Tigers_logo.png', out='temp.png', bar=None)
        im = iio.imread('temp.png')
        im_arr = np.array(im)
        os.remove('temp.png')

        # Assume that blank values are zeros...
        im_arr = np.mean(im_arr, axis=-1)

        # Reduce resolution by keeping every 10th point.
        data = []
        for row in range(im_arr.shape[0]):
            for col in range(im_arr.shape[1]):
                if row % 5 == 0 and col % 5 == 0:
                    entry = (col, -row, im_arr[row][col])
                    data.append(entry)
        data = np.array(data)

        # Convert to classification task.
        data[:,-1] = np.where(data[:,-1] > 0.01, 1.0, -1.0)

        # Rescale axes.
        data[:,0] = (data[:,0] - np.min(data[:,0])) / (np.max(data[:,0]) - np.min(data[:,0]))
        data[:,1] = (data[:,1] - np.min(data[:,1])) / (np.max(data[:,1]) - np.min(data[:,1]))

        # Visualize dataset.
        if args.visualize:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            plt.rcParams['axes.labelsize'] = 14
            negative = '#17233F'
            positive = '#DE6006'
            plt.figure(figsize=(7,7))
            princeton_cmap = colors.ListedColormap([negative, positive], name='classify_colormap', N=len([-1,1]))
            plt.scatter(data[:,0], data[:,1], s=8.0, c=data[:,2], cmap=princeton_cmap)
            plt.xlabel(r'$\phi_{1}$')
            plt.ylabel(r'$\phi_{2}$')
            plt.xlim(xmin=0.0, xmax=1.0)
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.show()

        # Display dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/princeton.npy', data)

    # Generate the Glotzer phase diagram for constant packing fraction.
    if args.dataset == 'glotzer_pf':

        # Set bounds and resolution for dataset. Set to 10,000 points with the
        # bounds used in the paper.
        xmin = 0.0
        xmax = 500.0
        xnum = 100
        ymin = 0.0
        ymax = 1.0
        ynum = 100

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
        np.save('Datasets/glotzer_pf.npy', data)

    # Generate the Glotzer phase diagram for constant fraction of active particles.
    if args.dataset == 'glotzer_xa':

        # Set bounds and resolution for dataset. Set to 10,000 points with the
        # bounds used in the paper.
        xmin = 0.0
        xmax = 500.0
        xnum = 100
        ymin = 0.0
        ymax = 0.8
        ynum = 100

        # Create grid with desired resolution. 
        xs = np.linspace(xmin, xmax, xnum)
        ys = np.linspace(ymin, ymax, ynum)
        coords = np.array([(x, y) for x in xs for y in ys])
        labels = np.zeros((xnum * ynum, 1))
        data = np.hstack((coords, labels))

        # Assign labels, assuming x_A = 0.5.
        constant = 3.0 * math.pi * math.pi * 4.05 / (0.5 * 4.0)
        for row in range(data.shape[0]):
            if data[row, 0] > 225 and data[row, 1] > 0.425:
                data[row, 2] = 1
            elif data[row, 0] > 125 and data[row, 0] <= 225 and data[row, 1] > 0.4375:
                data[row, 2] = 1
            elif data[row, 0] > 0 and data[row, 0] <= 125 and data[row, 1] > constant / data[row, 0]:
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
            plt.ylabel('Packing Fraction')
            plt.xlim(xmin=xmin, xmax=xmax)
            plt.ylim(ymin=ymin, ymax=ymax)
            plt.show()

        # Report shape of dataset.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/glotzer_xa.npy', data)

    # Generate phase diagram for water at low pressure.
    if args.dataset == 'water_lp':

        # Define water phase diagram properties.
        sublimation_line = [
            [190, 3.22629E-07],
            [195, 7.38491E-07],
            [200, 1.62265E-06],
            [205, 3.43245E-06],
            [210, 7.00849E-06],
            [215, 1.38461E-05],
            [220, 2.65258E-05],
            [225, 4.93755E-05],
            [230, 8.94653E-05],
            [235, 1.58062E-04],
            [240, 2.72711E-04],
            [245, 4.60146E-04],
            [250, 7.60290E-04],
            [255, 1.23163E-03],
            [260, 1.95831E-03],
            [265, 3.05942E-03],
            [270, 4.70078E-03],
            [273.16, 6.11657E-03]
        ]

        melting_line = [
            [251.165, 2098.984998],
            [252, 2035.356888],
            [253, 1958.643733],
            [254, 1881.245432],
            [255, 1803.019346],
            [256, 1723.813966],
            [257, 1643.468248],
            [258, 1561.810917],
            [259, 1478.659711],
            [260, 1393.820594],
            [261, 1307.086907],
            [262, 1218.238469],
            [263, 1127.040619],
            [264, 1033.243204],
            [265, 936.5794978],
            [266, 836.765048],
            [267, 733.496462],
            [268, 626.4501074],
            [269, 515.2807366],
            [270, 399.6200239],
            [271, 279.0750131],
            [272, 153.2264695],
            [273, 21.62712952],
            [273.16, 0.00611657]
        ]

        saturation_line = [
            [273.16, 0.00611657],
            [275, 0.006984535],
            [280, 0.009918164],
            [285, 0.013890056],
            [290, 0.019199333],
            [295, 0.026211058],
            [300, 0.035365894],
            [305, 0.047190247],
            [310, 0.062306792],
            [315, 0.081445262],
            [320, 0.10545337],
            [325, 0.135307748],
            [330, 0.172124756],
            [335, 0.217171055],
            [340, 0.271873823],
            [345, 0.337830503],
            [350, 0.416818004],
            [355, 0.510801262],
            [360, 0.621941099],
            [365, 0.752601337],
            [370, 0.90535512],
            [375, 1.082990428],
            [380, 1.28851478],
            [385, 1.525159108],
            [390, 1.796380846],
            [395, 2.105866229],
            [400, 2.457531863],
            [405, 2.855525581],
            [410, 3.304226647],
            [415, 3.808245358],
            [420, 4.372422097],
            [425, 5.001825897],
            [430, 5.701752585],
            [435, 6.477722564],
            [440, 7.335478307],
            [445, 8.280981625],
            [450, 9.320410791],
            [455, 10.46015758],
            [460, 11.70682433],
            [465, 13.06722102],
            [470, 14.54836262],
            [475, 16.15746656],
            [480, 17.90195063],
            [485, 19.78943128],
            [490, 21.82772245],
            [495, 24.02483511],
            [500, 26.38897756],
            [505, 28.92855667],
            [510, 31.65218023],
            [515, 34.56866065],
            [520, 37.68702009],
            [525, 41.01649749],
            [530, 44.5665576],
            [535, 48.34690257],
            [540, 52.36748632],
            [545, 56.6385325],
            [550, 61.1705564],
            [555, 65.97439181],
            [560, 71.06122377],
            [565, 76.44262826],
            [570, 82.13062065],
            [575, 88.13771459],
            [580, 94.47699391],
            [585, 101.1622007],
            [590, 108.2078438],
            [595, 115.6293328],
            [600, 123.4431458],
            [605, 131.6670403],
            [610, 140.320322],
            [615, 149.4241944],
            [620, 159.0022218],
            [625, 169.0809733],
            [630, 179.6909846],
            [635, 190.8684489],
            [640, 202.6594217],
            [645, 215.1413929],
            [647.096, 220.64]
        ]

        sublimation_line = np.array(sublimation_line)
        melting_line = np.array(melting_line)
        saturation_line = np.array(saturation_line)

        sublimation_line[:,0] -= 273.15
        melting_line[:,0] -= 273.15
        saturation_line[:,0] -= 273.15

        # Generate water phase diagram for low pressures.
        x_min = -300
        x_max = 400
        x_res = 25
        y_min = -5
        y_max = 3
        y_res = 25
        x_vals = np.linspace(x_min, x_max, x_res)
        y_vals = np.logspace(y_min, y_max, y_res)
        xx, yy = np.meshgrid(x_vals, y_vals)

        # Classify every point based on the appropriate lines.
        inputs = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
        labels = np.zeros((x_vals.shape[0] * y_vals.shape[0], 1))
        for q in range(inputs.shape[0]):
            row = inputs[q]

            # Check if pressure is greater than triple point pressure.
            if row[1] > 0.006611657:

                # Check to see if temperature value is between melting and saturation line.
                melting_temp = 0.0
                for id, val in enumerate(melting_line):
                    if val[1] < row[1]:
                        T1 = melting_line[id-1][0]
                        T2 = melting_line[id][0]
                        P1 = melting_line[id-1][1]
                        P2 = melting_line[id][1]

                        # Linearly interpolate the cutoff temperature value.
                        factor = (row[1] - P1) / (P2 - P1)
                        melting_temp = factor * (T2 - T1) + T1

                        break

                saturation_temp = 100000.0
                for id, val in enumerate(saturation_line):
                    if val[1] > row[1]:
                        T1 = saturation_line[id-1][0]
                        T2 = saturation_line[id][0]
                        P1 = saturation_line[id-1][1]
                        P2 = saturation_line[id][1]

                        # Linearly interpolate the cutoff temperature value.
                        factor = (row[1] - P1) / (P2 - P1)
                        saturation_temp = factor * (T2 - T1) + T1

                        break

                if row[0] > melting_temp and row[0] < saturation_temp:
                    labels[q, 0] = 1

        # Aggregate data into dataset.
        labels = np.where(labels != 1, -1, 1)
        data = np.hstack((inputs, labels))

        # Since the x_2 variables are spaced out on a log scale, take the log of this value.
        data[:,1] = np.log(data[:,1])

        # Visualize dataset, if specified.
        if args.visualize:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(7,7))
            plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=plt.get_cmap('bwr'), s=10.0)
            plt.xlabel(r'$T$ [C]')
            plt.ylabel(r'$P$ [MPa] (Log)')
            plt.xlim(xmin=x_min, xmax=x_max)
            plt.ylim(ymin=y_min, ymax=y_max)
            plt.show()

        # Report shape of dataset.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/water_lp.npy', data)

    # Construct phase diagram for water at high pressures.
    if args.dataset == 'water_hp':

        # Define water phase diagram properties.
        sublimation_line = [
            [190, 3.22629E-07],
            [195, 7.38491E-07],
            [200, 1.62265E-06],
            [205, 3.43245E-06],
            [210, 7.00849E-06],
            [215, 1.38461E-05],
            [220, 2.65258E-05],
            [225, 4.93755E-05],
            [230, 8.94653E-05],
            [235, 1.58062E-04],
            [240, 2.72711E-04],
            [245, 4.60146E-04],
            [250, 7.60290E-04],
            [255, 1.23163E-03],
            [260, 1.95831E-03],
            [265, 3.05942E-03],
            [270, 4.70078E-03],
            [273.16, 6.11657E-03]
        ]

        melting_line = [
            [251.165, 2098.984998],
            [252, 2035.356888],
            [253, 1958.643733],
            [254, 1881.245432],
            [255, 1803.019346],
            [256, 1723.813966],
            [257, 1643.468248],
            [258, 1561.810917],
            [259, 1478.659711],
            [260, 1393.820594],
            [261, 1307.086907],
            [262, 1218.238469],
            [263, 1127.040619],
            [264, 1033.243204],
            [265, 936.5794978],
            [266, 836.765048],
            [267, 733.496462],
            [268, 626.4501074],
            [269, 515.2807366],
            [270, 399.6200239],
            [271, 279.0750131],
            [272, 153.2264695],
            [273, 21.62712952],
            [273.16, 0.00611657]
        ]

        saturation_line = [
            [273.16, 0.00611657],
            [275, 0.006984535],
            [280, 0.009918164],
            [285, 0.013890056],
            [290, 0.019199333],
            [295, 0.026211058],
            [300, 0.035365894],
            [305, 0.047190247],
            [310, 0.062306792],
            [315, 0.081445262],
            [320, 0.10545337],
            [325, 0.135307748],
            [330, 0.172124756],
            [335, 0.217171055],
            [340, 0.271873823],
            [345, 0.337830503],
            [350, 0.416818004],
            [355, 0.510801262],
            [360, 0.621941099],
            [365, 0.752601337],
            [370, 0.90535512],
            [375, 1.082990428],
            [380, 1.28851478],
            [385, 1.525159108],
            [390, 1.796380846],
            [395, 2.105866229],
            [400, 2.457531863],
            [405, 2.855525581],
            [410, 3.304226647],
            [415, 3.808245358],
            [420, 4.372422097],
            [425, 5.001825897],
            [430, 5.701752585],
            [435, 6.477722564],
            [440, 7.335478307],
            [445, 8.280981625],
            [450, 9.320410791],
            [455, 10.46015758],
            [460, 11.70682433],
            [465, 13.06722102],
            [470, 14.54836262],
            [475, 16.15746656],
            [480, 17.90195063],
            [485, 19.78943128],
            [490, 21.82772245],
            [495, 24.02483511],
            [500, 26.38897756],
            [505, 28.92855667],
            [510, 31.65218023],
            [515, 34.56866065],
            [520, 37.68702009],
            [525, 41.01649749],
            [530, 44.5665576],
            [535, 48.34690257],
            [540, 52.36748632],
            [545, 56.6385325],
            [550, 61.1705564],
            [555, 65.97439181],
            [560, 71.06122377],
            [565, 76.44262826],
            [570, 82.13062065],
            [575, 88.13771459],
            [580, 94.47699391],
            [585, 101.1622007],
            [590, 108.2078438],
            [595, 115.6293328],
            [600, 123.4431458],
            [605, 131.6670403],
            [610, 140.320322],
            [615, 149.4241944],
            [620, 159.0022218],
            [625, 169.0809733],
            [630, 179.6909846],
            [635, 190.8684489],
            [640, 202.6594217],
            [645, 215.1413929],
            [647.096, 220.64]
        ]

        sublimation_line = np.array(sublimation_line)
        melting_line = np.array(melting_line)
        saturation_line = np.array(saturation_line)

        sublimation_line[:,0] -= 273.15
        melting_line[:,0] -= 273.15
        saturation_line[:,0] -= 273.15

        # Define ice line for high pressure water.
        ice_line = [
            [251.165, 2099],
            [251.500, 2150.598202],
            [252.000, 2235.559833],
            [252.500, 2331.072709],
            [253.000, 2438.422284],
            [253.500, 2559.047602],
            [254.000, 2694.559285],
            [254.500, 2846.75959],
            [255.000, 3017.664767],
            [255.500, 3209.529967],
            [256.000, 3424.876998],
            [256.164, 3501.092393],
            [256.164, 3501],
            [257, 3610.764657],
            [258, 3745.386945],
            [259, 3883.711755],
            [260, 4025.826027],
            [261, 4171.8184],
            [262, 4321.779232],
            [263, 4475.800629],
            [264, 4633.976472],
            [265, 4796.402444],
            [266, 4963.176056],
            [267, 5134.396677],
            [268, 5310.165563],
            [269, 5490.585883],
            [270, 5675.762747],
            [271, 5865.803239],
            [272, 6060.816446],
            [273, 6260.913482],
            [273.31, 6323.993474],
            [273.31, 6324],
            [274, 6403.291777],
            [276, 6637.2199],
            [278, 6877.330747],
            [280, 7123.741478],
            [282, 7376.57061],
            [284, 7635.938022],
            [286, 7901.964961],
            [288, 8174.77405],
            [290, 8454.489288],
            [292, 8741.236063],
            [294, 9035.141153],
            [296, 9336.332731],
            [298, 9644.940375],
            [300, 9961.095071],
            [302, 10284.92922],
            [304, 10616.57664],
            [306, 10956.17258],
            [308, 11303.85371],
            [310, 11659.75815],
            [312, 12024.02544],
            [314, 12396.79661],
            [316, 12778.2141],
            [318, 13168.42183],
            [320, 13567.56518],
            [322, 13975.791],
            [324, 14393.24762],
            [326, 14820.08486],
            [328, 15256.45399],
            [330, 15702.50782],
            [332, 16158.40063],
            [334, 16624.28821],
            [336, 17100.32786],
            [338, 17586.67838],
            [340, 18083.50012],
            [342, 18590.95493],
            [344, 19109.2062],
            [346, 19638.41885],
            [348, 20178.75934],
            [350, 20730.3957],
            [352, 21293.49748],
            [354, 21868.2358],
            [355, 22160.02257],
            [355, 22160],
            [360, 22790.78512],
            [365, 23429.31863],
            [370, 24076.12792],
            [375, 24731.78711],
            [380, 25396.91878],
            [385, 26072.19576],
            [390, 26758.34323],
            [395, 27456.14096],
            [400, 28166.4259],
            [405, 28890.09496],
            [410, 29628.10808],
            [415, 30381.49165],
            [420, 31151.34222],
            [425, 31938.83054],
            [430, 32745.2061],
            [435, 33571.80191],
            [440, 34420.03992],
            [445, 35291.43672],
            [450, 36187.60993],
            [455, 37110.28506],
            [460, 38061.30294],
            [465, 39042.62789],
            [470, 40056.3565],
            [475, 41104.72714],
            [480, 42190.13028],
            [485, 43315.11967],
            [490, 44482.42425],
            [495, 45694.96114],
            [500, 46955.84945],
            [505, 48268.42511],
            [510, 49636.25662],
            [515, 51063.16186],
            [520, 52553.22571],
            [525, 54110.81863],
            [530, 55740.61593],
            [535, 57447.6176],
            [540, 59237.16855],
            [545, 61114.97866],
            [550, 63087.14244],
            [555, 65160.15748],
            [560, 67340.94091],
            [565, 69636.84271],
            [570, 72055.65451],
            [575, 74605.61196],
            [580, 77295.38837],
            [585, 80134.07668],
            [590, 83131.15583],
            [595, 86296.43709],
            [600, 89639.98425],
            [605, 93172.00061],
            [610, 96902.67363],
            [615, 100841.9666],
            [620, 104999.3437],
            [625, 109383.4133],
            [630, 114001.4699],
            [635, 118858.9125],
            [640, 123958.5159],
            [645, 129299.5234],
            [650, 134876.5326],
            [655, 140678.1401],
            [660, 146685.3144],
            [665, 152869.4698],
            [670, 159190.2242],
            [675, 165592.842],
            [680, 172005.3942],
            [685, 178335.7103],
            [690, 184468.2654],
            [695, 190261.2344],
            [700, 195544.0625],
            [705, 200116.0506],
            [710, 203746.6236],
            [715, 206178.1282]
        ]

        ice_line = np.array(ice_line)
        ice_line[:,0] = ice_line[:,0] - 273.15

        # Generate water phase diagram for low pressures.
        x_min = -60
        x_max = 30
        x_res = 25
        y_min = 0
        y_max = 10000
        y_res = 25
        x = np.linspace(x_min, x_max, x_res)
        y = np.linspace(y_min, y_max, y_res)
        xx, yy = np.meshgrid(x, y)
        inputs = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
        labels = np.zeros((inputs.shape[0], 1))

        # Classify every point based on the appropriate lines.
        inputs = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
        labels = np.zeros((x_res * y_res, 1))
        for q in range(inputs.shape[0]):
            row = inputs[q]

            # Check if pressure is greater than triple point pressure.
            if row[1] > 0.006611657:

                # Check to see if temperature value is between melting and saturation line.
                melting_temp = 0.0
                if row[1] < 2099.0:
                    for id, val in enumerate(melting_line):
                        if val[1] < row[1]:
                            T1 = melting_line[id-1][0]
                            T2 = melting_line[id][0]
                            P1 = melting_line[id-1][1]
                            P2 = melting_line[id][1]

                            # Linearly interpolate the cutoff temperature value.
                            factor = (row[1] - P1) / (P2 - P1)
                            melting_temp = factor * (T2 - T1) + T1

                            break
                else:
                    for id, val in enumerate(ice_line):
                        if val[1] > row[1]:
                            T1 = ice_line[id-1][0]
                            T2 = ice_line[id][0]
                            P1 = ice_line[id-1][1]
                            P2 = ice_line[id][1]

                            # Linearly interpolate the cutoff temperature value.
                            factor = (row[1] - P1) / (P2 - P1)
                            melting_temp = factor * (T2 - T1) + T1
                            break


                saturation_temp = 100000.0
                for id, val in enumerate(saturation_line):
                    if val[1] > row[1]:
                        T1 = saturation_line[id-1][0]
                        T2 = saturation_line[id][0]
                        P1 = saturation_line[id-1][1]
                        P2 = saturation_line[id][1]

                        # Linearly interpolate the cutoff temperature value.
                        factor = (row[1] - P1) / (P2 - P1)
                        saturation_temp = factor * (T2 - T1) + T1

                        break

                if row[0] > melting_temp and row[0] < saturation_temp:
                    labels[q, 0] = 1

        # Aggregate data into dataset.
        labels = np.where(labels != 1, -1, 1)
        data = np.hstack((inputs, labels))

        # Visualize dataset, if specified.
        if args.visualize:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(7,7))
            plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=plt.get_cmap('bwr'), s=10.0)
            plt.xlabel(r'$T$ [C]')
            plt.ylabel(r'$P$ [MPa]')
            plt.xlim(xmin=x_min, xmax=x_max)
            plt.ylim(ymin=y_min, ymax=y_max)
            plt.show()

        # Report shape of dataset.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/water_hp.npy', data)

    # Construct oxidation boundary phase diagram.
    if args.dataset == 'oxidation':

        # Define the oxidation boundary polygon used in Bhat and Kitchin, IE&C 2023.
        polygon1 = Polygon([
            [0, 0.35],
            [0.05, 0.3],
            [0.1, 0.28],
            [0.15, 0.25],
            [0.2, 0.18],
            [0.25, 0.15],
            [0.3, 0.1],
            [0.35, 0.11],
            [0.4, 0.13],
            [0.45, 0.16],
            [0.5, 0.16],
            [0.55, 0.14],
            [0.6, 0.1],
            [0.65, 0.08],
            [0.7, 0],
            [0, 0]
            ])
        xi, yi = polygon1.exterior.xy

        # Turn the 2D polygon into 3D
        gridx, gridy, gridz = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        grid = np.stack([gridx, gridy], axis=-1)
        Xcon = np.array([gridx.flatten(), gridy.flatten(), gridz.flatten()]).T

        # Down select the mesh to include only the ternary values that add to 1
        Xcon = Xcon[(np.sum(Xcon, axis=1) < 1.0001) & (np.sum(Xcon, axis=1) > 0.999)]
        ds = dsam([0.7, 0.4], polygon1, boundmin=0, boundmax=1)

        # Classify the samples.
        cat = ds.classify_X(Xcon[:, :2])

        def trans3d(X):
            ''' Transforms a 2D space into 3D where all points within the
            set add to 1.
            X: 2D array where each row represents a point and the columns are x1, x2.
            '''
            x = np.array(X[:, 0])
            y = np.array(X[:, 1])
            z = np.array(1 - (x + y))
            return np.array([x, y, z]).T
        
        # Visualize tenary diagram.
        if args.visualize:
            plt.rcParams['font.family'] = 'Arial'
            fig, ax = plt.subplots(1, 1, figsize=(7,7))
            xpoly = trans3d(np.array([xi, yi]).T)
            td1 = TernaryDiagram(['Ni', 'Fe', 'Al'], ax=ax)
            td1.scatter(vector=Xcon, c=cat, alpha=0.8, s=10.0, cmap=plt.get_cmap('bwr'))
            td1.plot(vector=xpoly, c='red')
            plt.show()

        # Construct data np.array for saving.
        features = Xcon[:, 0:2]
        labels = np.array(cat).reshape(-1,1)
        labels = np.where(labels != 1, -1, 1)
        data = np.hstack((features, labels))

        # Display dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/oxidation.npy', data)

    # Construct solutions to the shower problem.
    if args.dataset == 'shower':

        # Generate solution boundary.
        polygon1 = Polygon([(0.4, 1.6), (1, 4), (1.7, 3.3), (0.7, 1.32)]) # input
        polygon2 = Polygon([(2, 98), (2, 105), (5, 105), (5, 98)]) # output
        xi, yi = polygon1.exterior.xy
        xo, yo = polygon2.exterior.xy
        gridx, gridy = np.meshgrid(np.linspace(0, 2, 25), np.linspace(0, 5, 25)) # enumerate for the control sampling
        X = np.array([gridx.flatten(), gridy.flatten()]).T
        ds = dsam([0.4, 1.6], polygon1, boundmin=0, boundmax=5)
        cat = ds.classify_X(X) # classify control
        bound = ds.get_bound(X, cat)

        if args.visualize:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_xlim((0, 2))
            ax.set_ylim((0, 5))
            ax.set_ylabel('Hot Water Volumetric Flowrate')
            ax.set_xlabel('Cold Water Volumetric Flowrate')
            ax.scatter(X[:, 0], X[:, 1], c=cat, s=10.0, cmap=plt.get_cmap('bwr'))
            plt.show()

        labels = np.where(np.array(cat) < 1, -1, 1).reshape(-1,1)
        data = np.hstack((X, labels))

        # Display dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/shower.npy', data)

    # Construct phase diagram for VDW equation of state.
    if args.dataset == 'vdw':

        def objective(pr, Tr):
            p = np.array([1, (-1 / 3 * (1 + 8 * Tr / pr))[0], (3 / pr)[0], (-1 / pr)[0]])
            r = np.sort(np.roots(p))
            LA = np.diff(r) * pr # area under horizontal line
            A = 8 * Tr * np.log(3 * r - 1) / 3 + 3 / r # area at each root
            RA = np.diff(A) - LA # real area between line and vdw
            return np.sum(RA) # the sum is zero (equal but opposite sign)

        def vdw(vr, tr):
            return 8 * tr / (3 * vr - 1) - 3 / vr**2
    
        from scipy.optimize import root
        Vr = np.linspace(0.5, 5)
        Tr = 0.85
        zr = np.array([x for x in np.roots([24 * Tr, -54, 36, -6]) if (x > 1 / 3)])
        pr_guess = np.mean(vdw(zr, Tr))
        sol = root(objective, pr_guess, args=(Tr,))
        pr = sol.x
        p = [1, (-1 / 3 * (1 + 8 * Tr / pr))[0], (3 / pr)[0], (-1 / pr)[0]]
        r = np.sort(np.roots(p))

        def phase(vr, pr, Tr):
            if Tr >= 1:
                return 0
            
            # Find phase boundary
            zr = np.array([x for x in np.roots([24 * Tr, -54, 36, -6]) if (x > 1 / 3)])
            pr_guess = np.mean(vdw(zr, Tr))
            sol = root(objective, pr_guess, args=(Tr,))
            vpr = sol.x
            p = [1, (-1 / 3 * (1 + 8 * Tr / vpr))[0], (3 / vpr)[0], (-1 / vpr)[0]]
            r = np.sort(np.roots(p))
            bounds = r[0], r[2]
            if (vr < r[0]) or (vr > r[2]):
                return 0
            else:
                return 1
        
        N = 200
        _Vr = np.linspace(0.5, 3.5, N)
        _pr = np.linspace(0.6, 1.2, N)
        Vr, pr = np.meshgrid(_Vr, _pr)
        Tr = 1 / 8 * (pr + 3 / Vr**2) * (3 * Vr - 1)
        cat = [phase(x, y, z) for x, y, z in zip(Vr.flatten(), pr.flatten(), Tr.flatten())]
        eosbound = plt.contour(Vr, pr, np.reshape(cat, Vr.shape), levels=[0.5])
        plt.plot(eosbound.allsegs[0][0][:, 0], eosbound.allsegs[0][0][:, 1])
        plt.xlim((0.5, 3))
        plt.xlabel('$V_r$')
        plt.ylabel('$p_r$')
        polygon1 = Polygon(eosbound.allsegs[0][0])
        xi, yi = polygon1.exterior.xy
        plt.close()

        gridx, gridy = np.meshgrid(np.linspace(0.5, 3, 25), np.linspace(0.6, 1.1, 25))
        X = np.array([gridx.flatten(), gridy.flatten()]).T
        ds = dsam([0.65, 0.65], polygon1, boundmin=0.5, boundmax=3)
        cat = ds.classify_X(X)

        data = np.hstack((X, np.where(np.array(cat) < 1, -1, 1).reshape(-1,1)))
        features = data[:, 0:-1]
    
        if args.visualize:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.scatter(X[:, 0], X[:, 1], c=cat, s=10.0, cmap=plt.get_cmap('bwr'))
            ax.set_xlim((0.45, 3.05))
            ax.set_ylim((0.59, 1.11))
            ax.set_xlabel('$V_r$')
            ax.set_ylabel('$p_r$')
            plt.show()

        # Display dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/vdw.npy', data)
    
    # Construct phase diagram for diblock copolymers.
    if args.dataset == 'diblock':

        # Get diblock.csv from online database.
        df = pd.read_csv('https://raw.githubusercontent.com/coleygroup/polymer-chemprop-data/main/datasets/diblock-phases/diblock.csv', low_memory=False)

        # Get and process dataset from Arora et al.
        df.loc[:, 'phase1'] = df.loc[:, 'phase1'].str.strip()
        df.loc[:, 'phase2'] = df.loc[:, 'phase2'].str.strip()
        df.loc[:, 'name1'] = df.loc[:, 'name1'].str.strip()
        df.loc[:, 'name2'] = df.loc[:, 'name2'].str.strip()
        cols = ['T', 'f1', 'Mn', 'phase1']
        df = df.loc[:, cols]

        # Rearrange so that the dataset is binary for lammellar or not.
        df['lamellar'] = np.where(df['phase1'] == 'lamellar', 1, -1)
        cols = ['T', 'f1', 'Mn', 'lamellar']
        df = df.loc[:, cols]
        data = df.to_numpy()

        # Visualize dataset.
        if args.visualize:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(data[:,0], data[:,1], data[:,2], c=data[:,3], s=5.0, cmap=plt.get_cmap('bwr'))
            ax.set_xlabel('T')
            ax.set_ylabel('f1')
            ax.set_zlabel('Mn');
            plt.show()

        # Display dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/diblock.npy', data)

    # Prepare additive manufacturing parameters that produce optimal toughness.
    if args.dataset == 'bear':

        # Read in raw data.
        df = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_crossed_barrel/raw_data.csv')

        # Determine threshold for converting regression to classification task.
        df['tough'] = np.where(df['toughness'] > 25.6546329, 1, -1)
        cols = ['n', 'theta', 'r', 't', 'tough']
        df = df.loc[:, cols]
        data = df.to_numpy()

        # Display dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/bear.npy', data)

    # Prepare electrocatalysts with top stability metric.
    if args.dataset == 'electro':

        # Get raw data from Github repository.
        df = pd.read_csv('https://raw.githubusercontent.com/masanorikodera/dd/main/dataset.csv')
        dataset = df.to_numpy()

        # Filter out NaN values.
        for row in range(dataset.shape[0]):
            if row < dataset.shape[0] and np.isnan(dataset[row][-1]):
                dataset = np.delete(dataset, row, axis=0)
                row -= 1

        # Get features and labels.
        features = dataset[:,1:5]
        labels = dataset[:,-1]

        # Choose classification labels.
        threshold = np.percentile(labels, q=0.80, axis=0)
        classes = np.where(labels > threshold, 1, -1).reshape(-1,1)
        data = np.hstack((features, classes))

        # Report results.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/electro.npy', data)

    # Task for samples with bottom 20% photodegradation.
    if args.dataset == 'hplc':

        # Get file from Olympus suite.
        df = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_hplc/data.csv')
        data = df.to_numpy()
        features = data[:,1:-1]

        # Determine threshold for converting from regression to classification.
        threshold = np.percentile(data[:,-1], q=20.0, axis=0)

        # Construct dataset, where positive labels are those samples with the
        # bottom 20% of photodegradation.
        labels = np.where(data[:,-1] < threshold, 1, -1).reshape(-1,1)
        data = np.hstack((features, labels))

        # Report results.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')
            
        # Save dataset.    
        np.save('Datasets/hplc.npy', data)

    # OER catalysts with minimal overpotentials.
    if args.dataset == 'oer':

        # Get raw data.
        df1 = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_oer_plate_3496/data.csv', header=None)
        df2 = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_oer_plate_3851/data.csv', header=None)
        df3 = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_oer_plate_3860/data.csv', header=None)
        df4 = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_oer_plate_4098/data.csv', header=None)

        # Convert to numpy format.
        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        df = df.groupby([0, 1, 2, 3, 4, 5]).mean().reset_index()

        # Generate data for threshold determination.
        # Goal: Minimize overpotential. Classify in the bottom 20% of overpotentials.
        overpotentials = df[6].to_numpy()
        threshold = np.percentile(overpotentials, q=20, axis=0)

        # Generate labels based on choice of threshold.
        df['satisfactory'] = np.where(df[6] < threshold, 1, -1)
        cols = [0, 1, 2, 3, 4, 5, 'satisfactory']
        df = df.loc[:, cols]
        data = df.to_numpy()

        # Report results.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/oer.npy', data)

    # Classifying radius of gyration from a polymer topology embedding.
    # Dataset generated in-house.
    if args.dataset == 'toporg':
        
        # Read in data from pickle file.
        datasets = []
        with open('Support/polymer_latent_space_with_label.pickle', 'rb') as handle:
            for _ in range(9):
                temp = pickle.load(handle)
                datasets.append(temp)

        # Aggregate features together.
        features = np.vstack((datasets[0], datasets[1], datasets[2]))
        rgs = np.hstack((datasets[3], datasets[4], datasets[5]))

        # Threshold for radius of gyration.
        threshold = 21.00
        labels = np.where(rgs < threshold, 1, -1)

        # Create data numpy array.
        data = np.hstack((features, labels.reshape(-1,1)))

        # Report results.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/toporg.npy', data)

    # Classifying polymers predicted to be gel-forming in a DMSO solution.
    # Dataset generated in-house.
    if args.dataset == 'polygel':

        # Load data.
        data = np.loadtxt('Support/gelation.dat')

        # Report results.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/polygel.npy', data)

    # Solubility of polymers in the specified solvent at room temperature.
    if args.dataset == 'polysol':

        # Get raw data.
        URL = 'https://pppdb.uchicago.edu/cloud_points'
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find(id='mastertemptable')
        data_points = results.find_all('tr')

        results = []
        for id, datum in enumerate(data_points):
            if id == 0:
                td_elements = datum.find_all('th')
                td_texts = [td.get_text(strip=True) for td in td_elements]
                results.append(td_texts)
            else:
                td_elements = datum.find_all('td')
                td_texts = [td.get_text(strip=True) for td in td_elements]
                results.append(td_texts)

        # Create dataframe.
        df = pd.DataFrame(results)
        header = df.iloc[0]
        df = df[1:]
        df.columns = header

        # Process raw data using Hansen solubility parameters.
        cols = ['Polymer', 'Solvent', 'Mw (Da)', 'PDI', 'ϕ', 'P (MPa)', '1-Phase', 'CP (°C)']
        df = df.loc[:, cols]
        df['Direction'] = np.where(df['1-Phase'] == 'positive', 1, 0)

        # Read in Hansen solubility parameters for polymer features.
        polymer_hansen = {}
        with open('Support/polymer_hansen.info', 'r') as param_file:
            lines = param_file.readlines()
            lines.pop(0)
            for i in range(int(len(lines) / 7)):
                section = lines[i*7:(i+1)*7]
                name = section[0]
                if name[-2] == '*':
                    name = name[0:-2]
                else:
                    name = name[0:-1]
                d1 = float(section[4])
                d2 = float(section[5])
                d3 = float(section[6].split()[0])
                polymer_hansen[name] = [d1, d2, d3]

        # Read in Hansen solubility parameters for solvent features.
        solvent_hansen = {}
        solv_df = pd.read_csv('Support/solvent_hansen.info')
        solv_df = solv_df[3:]
        solv_df.columns = ['Name', 'CAS', 'SMILES', 'rho', 'dD', 'dP', 'dH', '# cloud points']
        for index, row in solv_df.iterrows():
            row['Name'] = row['Name'].replace(' ', '')
        for index, row in solv_df.iterrows():
            solvent_hansen[row['Name']] = [float(row['dD']), float(row['dP']), float(row['dH'])]

        # Edit dataset to have appropriate descriptors.
        def label_polymer(row, val):
            return polymer_hansen[row['Polymer']][val]
        def label_solvent(row, val):
            return solvent_hansen[row['Solvent']][val]
        df['Solvent'] = df['Solvent'].str.replace(' ', '')
        df['polymer_dD'] = df.apply(lambda row: label_polymer(row, 0), axis=1)
        df['polymer_dP'] = df.apply(lambda row: label_polymer(row, 1), axis=1)
        df['polymer_dH'] = df.apply(lambda row: label_polymer(row, 2), axis=1)
        df['solv_dD'] = df.apply(lambda row: label_solvent(row, 0), axis=1)
        df['solv_dP'] = df.apply(lambda row: label_solvent(row, 1), axis=1)
        df['solv_dH'] = df.apply(lambda row: label_solvent(row, 2), axis=1)

        # Reorient dataframe in appropriate order.
        cols = ['polymer_dD', 'polymer_dP', 'polymer_dH', 'solv_dD', 'solv_dP', 
                'solv_dH', 'Mw (Da)', 'PDI', 'ϕ', 'P (MPa)', 'Direction', 'CP (°C)']
        df = df.loc[:, cols]

        # Decide on a threshold for classification.
        # Goal: To determine aggregation behavior at room temperature.
        data = df.to_numpy()
        cps = data[:,-1].astype(np.float64)
        good_cps = np.where(cps > 37.0, 1, -1)
        data = np.hstack((data[:,0:-1], good_cps.reshape(-1,1))).astype(np.float64)

        # Report results.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/polysol.npy', data)

    # Identify perovskites that are stable.
    if args.dataset == 'perovskite':

        # Get the original data file.
        df = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/atlas-unknown-constraints/main/application_hoip/ucb/reference-and-data/df_results.csv')

        # Get the precomputed descriptors.
        wget.download('https://github.com/aspuru-guzik-group/atlas-unknown-constraints/raw/main/application_hoip/ucb/reference-and-data/descriptors/desc_halogens.pkl', bar=None)
        wget.download('https://github.com/aspuru-guzik-group/atlas-unknown-constraints/raw/main/application_hoip/ucb/reference-and-data/descriptors/desc_metals.pkl', bar=None)
        wget.download('https://github.com/aspuru-guzik-group/atlas-unknown-constraints/raw/main/application_hoip/ucb/reference-and-data/descriptors/desc_molcats.pkl', bar=None)
            
        # Read in data.
        desc_halogens = pickle.load(open('desc_halogens.pkl', 'rb'))
        desc_metals = pickle.load(open('desc_metals.pkl', 'rb'))
        desc_molcats = pickle.load(open('desc_molcats.pkl', 'rb'))
        os.remove('desc_halogens.pkl')
        os.remove('desc_metals.pkl')
        os.remove('desc_molcats.pkl')

        # Define helper function for re-featurizing the dataset.
        def get_descriptors(component, desc):
            ''' generate a list of descritpors for the particular component
            '''
            desc_vec = []
            for key in desc.keys():
                desc_vec.append(desc[key][component])
            return list(np.array(desc_vec).astype(np.float64))

        def make_dataset(df_lookup, desc_halogens, desc_metals, desc_molcats):
            ''' generates the entire dataset
            '''
            desc_halogen_names = ['-'.join(['halogen', key]) for key in desc_halogens.keys()]
            desc_metal_names = ['-'.join(['metals', key]) for key in desc_metals.keys()]
            desc_molcat_names = ['-'.join(['molcats', key]) for key in desc_molcats.keys()]
            
            # is_feas = 0 (1) if (in)feasible
            all_data = {'molcat': [], 'metal': [], 'halogen': [], 'is_feas': [], 'bandgap_obj': [], 'm_star_obj': []}
            for desc_name in desc_molcat_names+desc_metal_names+desc_halogen_names:
                all_data[desc_name] = []
                
            molcats = ['H3S', 'NH4', 'MS', 'MA', 'MP', 'FA', 'EA', 'G', 'AA', 'ED', 'tBA']
            metals = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag',
                    'Au', 'Zn', 'Cd', 'Hg', 'Ga', 'In', 'Tl', 'Si', 'Ge', 'Sn', 'Pb', 'Bi', 'S', 'Se', 'Te']
            halogens = ['F', 'Cl', 'Br', 'I']

                
            for molcat in molcats:
                for metal in metals:
                    for halogen in halogens:
                        match = df_lookup.loc[(df_lookup['molcat'] == molcat) &
                                (df_lookup['metal'] == metal) &
                                (df_lookup['halogen'] == halogen)]
                        assert len(match) in [1, 0]
                        if len(match)==0:
                            bandgap_obj = np.nan
                            m_star_obj = np.nan
                            all_data['is_feas'].append(1)
                            
                        elif len(match) == 1:
                            bandgap_obj = np.abs(match.loc[:, 'bandgap'].to_numpy()[0] - 1.25)
                            m_star_obj = match.loc[:, 'm_star'].to_numpy()[0]
                            all_data['is_feas'].append(0)

                        else:
                            raise ValueError()

                        desc_molcat = get_descriptors(molcat, desc_molcats)
                        desc_metal = get_descriptors(metal, desc_metals)
                        desc_halogen = get_descriptors(halogen, desc_halogens)

                        all_data['molcat'].append(molcat)
                        all_data['metal'].append(metal)
                        all_data['halogen'].append(halogen)

                        all_data['bandgap_obj'].append(bandgap_obj)
                        all_data['m_star_obj'].append(m_star_obj)

                        for key, val in zip(desc_molcat_names, desc_molcat):
                            all_data[key].append(val)

                        for key, val in zip(desc_metal_names, desc_metal):
                            all_data[key].append(val)

                        for key, val in zip(desc_halogen_names, desc_halogen):
                            all_data[key].append(val )
                
            
            return pd.DataFrame(all_data)
        
        # Re-featurize dataset.
        df = make_dataset(df, desc_halogens, desc_metals, desc_molcats)

        # Choose desired columns, convert to numpy array and save.
        chosen_columns = ['molcats-scf_e', 'molcats-homo_e', 'molcats-lumo_e', 
                    'molcats-dip_mom_norm', 'molcats-radius_2d', 'molcats-mw', 'metals-electron_affinity', 'metals-electronegativity',
                    'metals-ionization_energy', 'metals-total_mass', 'halogen-electron_affinity', 'halogen-electronegativity', 
                    'halogen-ionization_energy', 'halogen-total_mass', 'is_feas']
        df_filt = df[chosen_columns]
        data = df_filt.to_numpy()

        # Set class 0 to -1.
        data[:,-1] = np.where(data[:,-1] < 1, -1, 1)

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save('Datasets/perovskite.npy', data)

    # Classifying molecules with the lowest band gaps from QM9.
    if 'qm9' in args.dataset:

        # Get the appropriate property.
        prop = args.dataset.split('_')[1]

        # Get original QM9 dataset.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv')

        # Downselect from the original QM9 dataset.
        sorted_dataset = df.sort_values(by=f'{prop}')
        sorted_dataset = sorted_dataset.reset_index(drop=True)
        reduced_dataset = sorted_dataset.iloc[::20]
        y = reduced_dataset[f'{prop}'].to_numpy()

        # Get Mordred descriptors for selected SMILES strings.
        # Note: This will take ~4 minutes to complete.
        smiles = reduced_dataset['smiles']
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols)
        X = df.to_numpy().astype(np.float64)

        # Process remaining features.
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Find those features (top 10) with largest coefficient 
        # values for a linear model.
        model = LassoCV()
        model.fit(df_scaled,y)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        threshold = np.percentile(y, q=20)
        labels_final = np.where(y < threshold, 1, -1).reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/qm9_{prop}.npy', data)

    # Identifying whether membranes made from repeat units are above
    # the 1999 Robeson bound for CO2/CH4 separation.
    if args.dataset == 'robeson':

        # Download raw data.
        df = pd.read_csv('https://raw.githubusercontent.com/jsunn-y/PolymerGasMembraneML/main/datasets/datasetA_imputed_all.csv')

        # Get relevant properties from dataset.
        df = df[['Smiles', 'log10_CO2_Bayesian', 'log10_CH4_Bayesian']]
        df['alpha_Bayesian'] = np.power(10, df['log10_CO2_Bayesian']) / np.power(10, df['log10_CH4_Bayesian'])
        df['log10_alpha_Bayesian'] = np.log10(df['alpha_Bayesian'])
        df = df.dropna(axis=0).reset_index()
        dataset = df.groupby('Smiles').max().reset_index()

        # Robeson bound values for CO2 / CH4.
        robeson_2008 = np.array([[-2.0, 3.311580178], [5.0, 0.65560772631]])
        robeson_1991 = np.array([[-0.3, 2.146128], [4.26, 0.636487]])

        # Get labels for each point.
        robeson = robeson_1991
        slope = (robeson[1,1] - robeson[0,1]) / (robeson[1,0] - robeson[0,0])
        y = np.where(dataset['log10_alpha_Bayesian'] > slope * (dataset['log10_CO2_Bayesian'] - robeson[0,0]) + robeson[0,1], 1, -1)

        # Convert SMILES strings to Mordred descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in dataset['Smiles']]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        y = y[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,y)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = y.reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/robeson.npy', data)

    # Classifying molecules with low free energies of solvation.
    if args.dataset == 'free':

        # Download raw data.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv')

        # Get relevant properties from dataset.
        smiles = df['smiles']
        y = df['expt']
        threshold = np.percentile(y, q=20)
        labels = np.where(y < threshold, 1, -1)

        # Convert SMILES strings to Mordred descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        y = y[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,labels)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = labels.reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/free.npy', data)

    # Classifying molecules with low solubility in water.
    if args.dataset == 'esol':

        # Download raw data.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv')

        # Get relevant properties from dataset.
        smiles = df['smiles']
        y = df['measured log solubility in mols per litre']
        threshold = np.percentile(y, q=20)
        labels = np.where(y < threshold, 1, -1)

        # Convert SMILES strings to Mordred descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        y = y[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,labels)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = labels.reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/esol.npy', data)

    # Classifying molecules with low experimental lipophilicity.
    if args.dataset == 'lipo':

        # Download raw data.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv')

        # Get relevant properties from dataset.
        smiles = df['smiles']
        y = df['exp']
        threshold = np.percentile(y, q=20)
        labels = np.where(y < threshold, 1, -1)

        # Convert SMILES strings to Mordred descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        y = y[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,labels)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = labels.reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/lipo.npy', data)

    # Classifying small molecule drugs as active against HIV or not.
    # Molecules are down-selected from the HIV MoleculeNet dataset.
    if args.dataset == 'hiv':

        # Get raw dataset.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv')

        # Get relevant properties from dataset.
        y = df['HIV_active'].to_numpy()
        inactive_rows = list(df.index[y == 0])
        active_rows = df.index[y == 1]

        # Make the final dataset 20% active, 80% inactive.
        rng = random.Random(12345)
        selected_inactive_indices = rng.sample(inactive_rows, k=1443*4)
        selected_active_indices = active_rows
        all_indices = []
        for index in selected_inactive_indices:
            all_indices.append(index)
        for index in selected_active_indices:
            all_indices.append(index)

        # Create new dataset and labels with the reduced dataset.
        df = df.iloc[all_indices]
        y = y[all_indices]
        smiles = df['smiles']
        labels = np.where(y == 1, 1, -1)

        # Convert SMILES strings to Mordred descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        y = y[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,labels)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = labels.reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/hiv.npy', data)

    # Classifying small molecules as active against BACE-1.
    if args.dataset == 'bace':

        # Download raw data.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv')

        # Get relevant properties from dataset.
        smiles = df['mol']
        y = df['Class']
        labels = np.where(y == 1, 1, -1)

        # Convert SMILES strings to Mordred descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        y = y[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,labels)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = labels.reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/bace.npy', data)

    # Classifying small molecules as being non-toxic during clinical trials.
    if args.dataset == 'clintox':

        # Download raw data.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz')

        # Get relevant properties from dataset.
        smiles = df['smiles']
        y = df['FDA_APPROVED']
        labels = np.where(y == 1, 1, -1)

        # Convert SMILES strings to Mordred descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        labels = labels[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,labels)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = labels.reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/clintox.npy', data)

    # Classification of molecules as active against a toxicity assay.
    if args.dataset == 'muv':

        # Load in raw dataset.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz')
        df = df[['smiles', 'MUV-466']]
        df = df.fillna(1.0)

        # Downselect dataset to create approximately 80/20 class split.
        inactive_rows = list(df.index[df['MUV-466'] == 0.0])
        active_rows = list(df.index[df['MUV-466'] == 1.0])

        # Make the final dataset 20% active, 80% inactive.
        rng = random.Random(12345)
        selected_inactive_indices = rng.sample(inactive_rows, k=1000)
        selected_active_indices = rng.sample(active_rows, k=4000)
        all_indices = []
        for index in selected_inactive_indices:
            all_indices.append(index)
        for index in selected_active_indices:
            all_indices.append(index)
        df = df.iloc[all_indices]
        smiles = df['smiles']
        labels = df['MUV-466'].to_numpy()
        labels = np.where(labels == 1.0, -1.0, 1.0) # Make minority class the 1.0.

        # Convert SMILES strings to molecular descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        labels = labels[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,labels)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = labels.reshape(-1,1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/muv.npy', data)

    # Classification of molecules as active against another toxicity assay.
    if args.dataset == 'tox21':

        # Load in raw dataset.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz')
        df = df.fillna(1.0)
        smiles = df['smiles']
        labels = df['SR-ARE'].to_numpy() # Choose the task with the most balanced class distribution.

        # Convert SMILES strings to Mordred descriptors.
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        labels = labels[keep_indices]
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(mols_filtered)
        df_without_nan_columns = df.dropna(axis=1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
        df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
        df_scaled = df_scaled.dropna(axis=1)
        df_unscaled = df_without_nan_columns[df_scaled.columns]

        # Determine those features most predictive of the label.
        model = LogisticRegressionCV(penalty='l1', solver='saga')
        model.fit(df_scaled,labels)
        weights = np.abs(np.array(model.coef_))
        feature_indices = np.argsort(-weights)[0][:10]
        chosen_features = []
        for index in feature_indices:
            print(f'{model.feature_names_in_[index]}............{model.coef_[0, index]}')
            chosen_features.append(model.feature_names_in_[index])
        features_final = df_unscaled[chosen_features].to_numpy().astype(np.float64)
        labels_final = labels.reshape(-1,1)
        labels_final = np.where(labels_final == 0, -1, 1)
        data = np.hstack((features_final, labels_final))

        # Report dataset shape.
        print(f'Shape of features: {data[:,0:-1].shape}')
        print(f'Shape of labels: {data[:,-1].shape}')

        # Save dataset.
        np.save(f'Datasets/tox21.npy', data)
