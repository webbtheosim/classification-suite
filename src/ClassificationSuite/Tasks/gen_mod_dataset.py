import argparse
from mordred import Calculator, descriptors
import numpy as np
import pandas as pd
import random
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

    # Get user input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=[
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
        default='qm9_gap'
    )
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--feat', default='mordred')
    parser.add_argument('--size', type=int, default=-1)
    args = parser.parse_args()

    # Prepared modified qm9 datasets.
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
        if args.feat == 'mordred':

            # Compute Mordred descriptors for molecules.
            smiles = reduced_dataset['smiles']
            mols = [Chem.MolFromSmiles(smi) for smi in smiles]
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols)

            # Process remaining features.
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float32))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Prepare labels.
            threshold = np.percentile(y, q=20)
            labels_final = np.where(y < threshold, 1, -1).reshape(-1,1)

            # Save a version using all Mordred descriptors.
            features_all = df_unscaled.to_numpy().astype(np.float32)
            data = np.hstack((features_all, labels_final))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/qm9_{prop}_mordred_all.npy', data)

            # Compute feature importances.
            model = LassoCV()
            model.fit(df_scaled,y)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels_final))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/qm9_{prop}_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels_final))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/qm9_{prop}_mordred_20.npy', data)

        # Compute dataset using Morgan fingerprints.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            smiles = reduced_dataset['smiles']
            mols = [Chem.MolFromSmiles(smi) for smi in smiles]
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)

            # Prepare labels.
            print('Preparing labels...')
            threshold = np.percentile(y, q=20)
            labels_final = np.where(y < threshold, 1, -1).reshape(-1,1)

            # Prepare final dataset.
            data = np.hstack((fps, labels_final))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/qm9_{prop}_morgan.npy', data.astype(int))

        else:
            raise Exception('Invalid featurization strategy specified.')
        
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

        # Only keep those entries where there are valid molecule objects created.
        mols = [Chem.MolFromSmiles(smi) for smi in dataset['Smiles']]
        keep_indices = [index for index, value in enumerate(mols) if value is not None]
        mols_filtered = []
        for id in range(len(mols)):
            if id in keep_indices:
                mols_filtered.append(mols[id])
        y = y[keep_indices]

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), y.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/robeson_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled,y)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, y.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/robeson_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, y.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/robeson_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, y.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/robeson_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')

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
        labels = labels[keep_indices]

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/free_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled, labels)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/free_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/free_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, labels.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/free_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')

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
        labels = labels[keep_indices]

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/esol_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled, labels)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/esol_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/esol_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, labels.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/esol_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')

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
        labels = labels[keep_indices]

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/lipo_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled, labels)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/lipo_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/lipo_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, labels.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/lipo_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')

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
        labels = labels[keep_indices]

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/hiv_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled, labels)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/hiv_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/hiv_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, labels.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/hiv_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')

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
        labels = labels[keep_indices]

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/bace_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled, labels)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/bace_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/bace_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, labels.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/bace_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')

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

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/clintox_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled, labels)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/clintox_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/clintox_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols_filtered]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, labels.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/clintox_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')

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

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/muv_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled, labels)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/muv_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/muv_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, labels.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/muv_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')

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

        # Save Mordred descriptor versions of the dataset.
        if args.feat == 'mordred':

            # Compute Mordred descriptors.
            calc = Calculator(descriptors, ignore_3D=True)
            df = calc.pandas(mols_filtered)
            df_without_nan_columns = df.dropna(axis=1)
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df_without_nan_columns.to_numpy().astype(np.float64))
            df_scaled = pd.DataFrame(df_scaled, columns=df_without_nan_columns.columns)
            df_scaled = df_scaled.dropna(axis=1)
            df_unscaled = df_without_nan_columns[df_scaled.columns]

            # Save a version with all Mordred descriptors.
            data = np.hstack((df_unscaled.to_numpy().astype(np.float32), labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/tox21_mordred_all.npy', data)

            # Compute feature importances of Mordred descriptors.
            model = LogisticRegressionCV(penalty='l1', solver='saga')
            model.fit(df_scaled, labels)
            weights = np.abs(np.array(model.coef_))
            feature_indices = np.argsort(-weights)[0]

            # Save a version using 100 Mordred descriptors.
            chosen_features = []
            for index in range(100):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/tox21_mordred_100.npy', data)

            # Save a version using 20 Mordred descriptors.
            chosen_features = []
            for index in range(20):
                print(index, model.feature_names_in_[feature_indices[index]])
                chosen_features.append(model.feature_names_in_[feature_indices[index]])
            features_final = df_unscaled[chosen_features].to_numpy().astype(np.float32)
            data = np.hstack((features_final, labels.reshape(-1,1)))
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/tox21_mordred_20.npy', data)

        # Save Morgan fingerprint versions of the dataset.
        elif args.feat == 'morgan':

            # Prepare features.
            print('Generating features...')
            mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps = [mfpgen.GetFingerprint(m) for m in mols]
            fps = np.array(fps).astype(int)
            data = np.hstack((fps, labels.reshape(-1,1)))

            # Save dataset.
            print(f'Shape of features: {data[:,0:-1].shape}')
            print(f'Shape of labels: {data[:,-1].shape}')
            np.save(f'Datasets/Modified/tox21_morgan.npy', data.astype(int))

        else:
            raise Exception('Specified featurization strategy is invalid.')
        
