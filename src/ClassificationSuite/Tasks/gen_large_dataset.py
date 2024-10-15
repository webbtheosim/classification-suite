import argparse
import math
import matplotlib.pyplot as plt
from mordred import Calculator, descriptors
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler

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
        x_res = 317
        y_min = -5
        y_max = 3
        y_res = 317
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
        np.save('Datasets/Modified/water_lp_large.npy', data)

    # Classifying molecules with the lowest band gaps from QM9.
    if 'qm9' in args.dataset:

        # Get the appropriate property.
        prop = args.dataset.split('_')[1]

        # Get original QM9 dataset.
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv')
        y = df[f'{prop}'].to_numpy()

        # Get Mordred descriptors for selected SMILES strings.
        # Note: This will take ~4 minutes to complete.
        smiles = df['smiles']
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
        np.save(f'Datasets/Modified/qm9_{prop}_large.npy', data)