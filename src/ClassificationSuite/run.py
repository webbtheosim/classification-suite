import argparse
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
import time
import torch
from celluloid import Camera

from ClassificationSuite.Models import *
from ClassificationSuite.Models.utils import *
from ClassificationSuite.Samplers import sample
from ClassificationSuite.Tasks.utils import load_data, task_config

import functools
print = functools.partial(print, flush=True)

def run_active_learning(task, seed, sampler, model, visualize=False):
    '''
        Method for running an active learning protocol with the specified
        parameters.
    '''

    # Set all random seeds.
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    
    # Load dataset.
    dataset = load_data(task=task)
    np.random.shuffle(dataset)
    X_all = dataset[:,0:-1]
    y_all = dataset[:,-1]

    # Get task configuration.
    config = task_config(task=task)
    seed_size  = config[0]
    batch_size = config[1]
    rounds     = config[2]

    # Choose sample.
    chosen_indices = sample(name=sampler, domain=X_all, size=seed_size, seed=seed, task=task)
    chosen_x = X_all[chosen_indices,:]
    chosen_y = y_all[chosen_indices]

    # Initialize model.
    if 'ensemble' not in model:
        model = get_model(name=model, task=task)
    else:
        if model == 'ensemble_top':
            model = TopModelEnsemble(models=['nn', 'rf', 'gpc_ard'])
        if model == 'ensemble_averaging':
            model = AveragingEnsemble(models=['nn', 'rf', 'gpc_ard'])
        if model == 'ensemble_stacking':
            model =  StackingEnsemble(models=['nn', 'rf', 'gpc_ard'])
        if model == 'ensemble_arbitrating':
            model = ArbitratingEnsemble(models=['nn', 'rf', 'gpc_ard'])

    # Execute active learning loop.
    results = np.zeros((rounds+1, 4))
    for i in range(rounds+1):

        # Report progress.
        print(f'Computing for round {i} / {rounds}.')

        # Train model on measured data.
        model.load_data(X=chosen_x, y=chosen_y)
        model.train(cv=True)

        # Measure accuracy of the model.
        y_pred = model.classify(X_test=X_all)
        results[i,0] = i
        results[i,1] = balanced_accuracy_score(y_all, y_pred)
        results[i,2] = f1_score(y_all, y_pred, average='macro')
        results[i,3] = matthews_corrcoef(y_all, y_pred)
        print(f'Macro F1 for round {i}: {results[i,2]:.3f}')

        # Visualize, if specified.
        if visualize:
            visualize_model_output(
                dataset=dataset,
                chosen_points=chosen_x,
                y_pred=y_pred,
                y_acq=model.uncertainty(X_test=X_all),
                size=batch_size
            )

        # Acquire new points if this isn't the last round.
        if i < rounds:
            new_indices = model.recommend(
                domain=X_all,
                size=batch_size, 
                chosen_indices=chosen_indices
            )
            chosen_x = np.vstack((chosen_x, X_all[new_indices]))
            chosen_y = np.hstack((chosen_y, y_all[new_indices]))
            for index in new_indices:
                chosen_indices.append(index)

    return results

def run_space_filling(task, seed, sampler, model, visualize=False):
    '''
        Method for running a space filling protocol with the specified
        parameters.
    '''

    # Set all random seeds.
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    
    # Load dataset.
    dataset = load_data(task=task)
    np.random.shuffle(dataset)
    X_all = dataset[:,0:-1]
    y_all = dataset[:,-1]

    # Get task configuration.
    config = task_config(task=task)
    seed_size  = config[0]
    batch_size = config[1]
    rounds     = config[2]

    # Build space-filling designs.
    results = np.zeros((rounds+1, 4))
    for i in range(rounds+1):

        # Report progress.
        print(f'Computing for round {i} / {rounds}.')

        # Choose sample.
        budget = i * batch_size + seed_size
        chosen_indices = sample(name=sampler, domain=X_all, size=budget, seed=seed, task=task)
        chosen_x = X_all[chosen_indices,:]
        chosen_y = y_all[chosen_indices]

        # Initialize model.
        if 'ensemble' not in model:
            model_ = get_model(name=model, task=task)
        else:
            if model == 'ensemble_top':
                model_ = TopModelEnsemble(models=['nn', 'rf', 'gpc_ard'])
            if model_ == 'ensemble_averaging':
                model = AveragingEnsemble(models=['nn', 'rf', 'gpc_ard'])
            if model_ == 'ensemble_stacking':
                model =  StackingEnsemble(models=['nn', 'rf', 'gpc_ard'])
            if model_ == 'ensemble_arbitrating':
                model = ArbitratingEnsemble(models=['nn', 'rf', 'gpc_ard'])

        # Train model on measured data.
        model_.load_data(X=chosen_x, y=chosen_y)
        model_.train(cv=True)

        # Measure accuracy of the model.
        y_pred = model_.classify(X_test=X_all)
        results[i,0] = i
        results[i,1] = balanced_accuracy_score(y_all, y_pred)
        results[i,2] = f1_score(y_all, y_pred, average='macro')
        results[i,3] = matthews_corrcoef(y_all, y_pred)
        print(f'Macro F1 for round {i}: {results[i,2]:.3f}')

        # Visualize, if specified.
        if visualize:
            visualize_model_output(
                dataset=dataset,
                chosen_points=chosen_x,
                y_pred=y_pred,
                y_acq=model_.uncertainty(X_test=X_all)
            )

    return results

if __name__ == '__main__':

    # Get protocol input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme', 
        choices=['al', 'sf'], 
        default='al'
    )
    parser.add_argument('--task', default='princeton')
    parser.add_argument('--sampler', 
        choices=['random', 'maximin', 'medoids', 'max_entropy', 'vendi'], 
        default='random'
    )
    parser.add_argument('--model', 
        choices=['gpc', 'gpc_ard', 'gpr', 'gpr_ard', 'bkde', 'knn', 'lp', 'nn', 'rf', 'sv', 'xgb',
                 'ensemble_top', 'ensemble_averaging', 'ensemble_stacking', 'ensemble_arbitrating'], 
        default='gpr'
    )
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--results_dir', default='.')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # Execute protocol.
    if args.scheme == 'al':
        start = time.perf_counter()
        results = run_active_learning(
            task=args.task,
            seed=args.seed,
            sampler=args.sampler,
            model=args.model,
            visualize=args.visualize
        )
        end = time.perf_counter()
    if args.scheme == 'sf':
        start = time.perf_counter()
        results = run_space_filling(
            task=args.task,
            seed=args.seed,
            sampler=args.sampler,
            model=args.model,
            visualize=args.visualize
        )
        end = time.perf_counter()

    # Save raw results to appropriate location.
    results_file_str = f'{args.scheme}-{args.task}-{args.sampler}-{args.model}-{args.seed}'
    np.save(file=f'{args.results_dir}/{results_file_str}.npy', arr=results)
    print(f'Calculations for {results_file_str} completed in {(end-start):.3f} seconds.')
