import argparse
from joblib import Parallel, delayed
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from ClassificationSuite.Models import KNN
from ClassificationSuite.Samplers import sample
from ClassificationSuite.Tasks.utils import load_data

import functools
print = functools.partial(print, flush=True)

def run_baseline(task, seeds, cutoff_frac=1.0):
    '''
        Compute the performance of a naïve baseline algorithm on the
        specified task. Cutoff_frac stops the baseline data early.
    '''
    
    # Load dataset.
    dataset = load_data(task=task)
    np.random.shuffle(dataset)
    X_all = dataset[:,0:-1]
    y_all = dataset[:,-1]

    # Define range of relevant sizes.
    eval_max = int(X_all.shape[0] * cutoff_frac)

    # Gather results in parallel.
    results = Parallel(n_jobs=-1)(delayed(_evaluate)(X_all, y_all, seed, size) for seed in range(seeds) for size in range(1,eval_max))

    return np.array(results)

def _evaluate(X_all, y_all, seed, size):
    
    '''
        Helper method that evaluates random-KNN for exactly one size
        and one task. Used by run_baseline to gather data in parallel.
    '''

    # Set all random seeds.
    np.random.seed(seed=seed)

    # Report progress.
    if (size % 500) == 0:
        print(f'Computing for {size} / {X_all.shape[0]}')

    # Choose sample.
    chosen_indices = sample(name='random', domain=X_all, size=size, seed=seed)
    chosen_x = X_all[chosen_indices,:]
    chosen_y = y_all[chosen_indices]

    # Initialize model.
    model_ = KNeighborsClassifier(n_neighbors=1)

    # Scale data.
    sc = MinMaxScaler().fit(chosen_x)
    chosen_x = sc.transform(chosen_x)

    # Train model on measured data.
    model_.fit(X=chosen_x, y=chosen_y)

    # Measure accuracy of the model.
    y_pred = model_.predict(X=sc.transform(X_all))
    results = [
        size,
        balanced_accuracy_score(y_all, y_pred),
        f1_score(y_all, y_pred, average='macro'),
        matthews_corrcoef(y_all, y_pred)
    ]

    # Save results to the baseline.pkl 
    return results

if __name__ == '__main__':

    # Get user input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
        choices=['bace', 'bear', 'clintox', 'diblock', 'electro', 'esol', 'free', 'glotzer_pf', 
                 'glotzer_xa', 'hiv', 'hplc', 'lipo', 'muv', 'oer', 'oxidation', 'perovskite', 
                 'polygel', 'polysol', 'princeton', 'qm9_cv', 'qm9_gap', 'qm9_r2', 'qm9_u0', 
                 'qm9_zpve', 'robeson', 'shower', 'toporg', 'tox21', 'vdw', 'water_hp', 'water_lp'], 
        default='princeton'
    )
    parser.add_argument('--seeds', type=int, default=30)
    args = parser.parse_args()

    # Compute baseline results.
    results = run_baseline(task=args.task, seeds=args.seeds, cutoff_frac=1.0)

    # Save results to file.
    path = Path(__file__).parent
    if os.path.exists(f'{path}/baseline_results.pickle'):
        with open(f'{path}/baseline_results.pickle', 'rb') as handle:
            results_dict = pickle.load(handle)
    else:
        results_dict = {}
    results_dict[args.task] = results
    with open(f'{path}/baseline_results.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Calculations completed for {args.task}.')
