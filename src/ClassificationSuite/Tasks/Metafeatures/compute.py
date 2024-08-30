import argparse
import os
import pickle
from pymfe.mfe import MFE

from ClassificationSuite.Tasks.utils import load_data

if __name__ == '__main__':

    # Get task from the user.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
        choices=['bace', 'bear', 'clintox', 'diblock', 'electro', 'esol', 'free', 'glotzer_pf', 
                 'glotzer_xa', 'hiv', 'hplc', 'lipo', 'muv', 'oer', 'oxidation', 'perovskite', 
                 'polygel', 'polysol', 'princeton', 'qm9_cv', 'qm9_gap', 'qm9_r2', 'qm9_u0', 
                 'qm9_zpve', 'robeson', 'shower', 'toporg', 'tox21', 'vdw', 'water_hp', 'water_lp'], 
        default='princeton'
    )
    args = parser.parse_args()

    # Load dataset.
    dataset = load_data(task=args.task)
    X = dataset[:,0:-1]
    y = dataset[:,-1]

    # Compute meta-features for this dataset.
    mfe = MFE(groups='all', summary=['mean', 'sd', 'min', 'max'])
    mfe.fit(X,y)
    ft = mfe.extract()

    # Structure results in a dictionary.
    task_metafeatures = {k: v for k,v in zip(ft[0], ft[1])}

    # Generate metafeatures.pickle file if it doesn't already exist.
    if not os.path.exists('./metafeatures.pickle'):
        metafeatures = {}
        with open('metafeatures.pickle', 'wb') as handle:
            pickle.dump(metafeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save results to the metafeatures pickle file.
    with open('metafeatures.pickle', 'rb') as handle:
        metafeatures = pickle.load(handle)
    metafeatures[args.task] = task_metafeatures
    with open('metafeatures.pickle', 'wb') as handle:
        pickle.dump(metafeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)