from joblib import Parallel, delayed
import kmedoids
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform, jaccard
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

def random(domain, size, seed):
    ''' 
        Select a random sample from the domain. Returns
        the indices of points chosen from the provided 
        domain.
    '''

    # Set random seed.
    np.random.seed(seed=seed)

    # Choose points from the domain.
    indices = [i for i in range(domain.shape[0])]
    sample = np.random.choice(a=indices, size=size, replace=False).tolist()

    return sample

def maximin(domain, size, seed, metric='euclidean'):
    '''
        Select a sample that maximizes the minimum distance
        between any two points in the sample, subject to 
        a randomly chose initial point.
    '''

    # Redefine distances if dealing with Morgan fingerprints.
    if metric == 'jaccard':
        print('Sampling using jaccard distance!')

    # Set random seed.
    np.random.seed(seed=seed)

    # Scale the domain.
    domain = MinMaxScaler().fit_transform(domain)

    # Choose an initial point.
    sample = [np.random.randint(low=0, high=domain.shape[0])]

    # Greedily select the point which maximizes the minimum
    # distance to other points in the sample.
    for _ in range(size-1):

        # Find minimum distance to any point currently in sample.
        chosen_points = domain[sample, :]
        distances = cdist(chosen_points, domain, metric=metric)
        distances = np.array(distances)
        distances = np.min(distances, axis=0).reshape(-1,1)

        # Sort points by minimum distance, find the max of this set.
        new_id = np.argsort(-distances, axis=0)[0].item()

        # Add point to the sample.
        sample.append(new_id)

    return sample

def medoids(domain, size, seed, metric='euclidean'):
    '''
        Select a sample that minimizes the variance between
        all points in the dataset to a selected point.
    '''

    # Redefine distances if dealing with Morgan fingerprints.
    if metric == 'jaccard':
        print('Sampling using jaccard distance!')

    # Use entire dataset if dealing with low domain sizes.
    if domain.shape[0] < 30000:

        # Scale data.
        domain = MinMaxScaler().fit_transform(domain)

        # Fit k-medoids.
        k = kmedoids.KMedoids(
            n_clusters=size, 
            metric=metric, 
            random_state=seed
        )
        output = k.fit(X=domain, y=None)
        sample = output.medoid_indices_.tolist()

    # Otherwise, down-sample to a reasonable size.
    else:
        print('Down-sampling...')

        # Sample data.
        domain_idx = [i for i in range(domain.shape[0])]
        small_idx = np.random.choice(domain_idx, size=10000, replace=False)
        domain_small = domain[small_idx]
        domain_small = MinMaxScaler().fit_transform(domain_small)
        
        # Fit k-medoids.
        k = kmedoids.KMedoids(
            n_clusters=size, 
            metric=metric, 
            random_state=seed
        )
        output = k.fit(X=domain_small, y=None)
        sample_small = output.medoid_indices_

        # Get indices of original domain.
        sample = small_idx[sample_small].tolist()

    return sample

def max_entropy(domain, size, seed, neighbors=100, metric='euclidean'):
    '''
        Select a sample that maximizes the Shannon entropy
        of the selection by assuming each point is a Gaussian
        distribution with bandwidths determined by Silverman's
        rule.
    '''

    # Redefine distances if dealing with Morgan fingerprints.
    if metric == 'jaccard':
        print('Sampling using jaccard distance!')
    
    # Set random seed.
    np.random.seed(seed=seed)

    # If the domain size is sufficiently small.
    down_sample = False
    if domain.shape[0] > 30000:
        down_sample = True
        print('Down-sampling...')
        domain_idx = [i for i in range(domain.shape[0])]
        small_idx = np.random.choice(domain_idx, size=10000, replace=False)
        domain = domain[small_idx]

    # Scale dataset.
    domain = MinMaxScaler().fit_transform(domain)

    # Choose the initial point.
    sample = [np.random.randint(low=0, high=domain.shape[0])]

    if metric == 'euclidean':

        # Compute bandwidths along each dimension.
        stds = np.std(domain, axis=0)
        lower_quartile = np.percentile(domain, 25, axis=0)
        upper_quartile = np.percentile(domain, 75, axis=0)
        iqrs = upper_quartile - lower_quartile
        stack_ranges = np.vstack((stds, iqrs))
        metric = np.min(stack_ranges, axis=0)
        bandwidths = 0.9 * metric * domain.shape[0]**(-0.2)

        # Some properties do not have much variation, so metric
        # evaluates to 0. Add a 'jitter' value to deal with this
        # situation.
        bandwidths = bandwidths + 1e-50

        # Find neighbors to use for later computations.
        nbrs = NearestNeighbors(n_neighbors=neighbors).fit(domain)
        _, indices = nbrs.kneighbors(domain)

        # Compute contributions to score from total dataset.
        def compute_itrss(input, id):
            datum = domain[id].reshape(1,-1)
            d2 = np.sum(np.square(input - datum) / np.square(bandwidths), axis=1)
            metric = np.sum(np.exp(-d2 / 2.0), axis=0)
            return metric
        total = Parallel(n_jobs=-1)(delayed(compute_itrss)(domain[indices[id]], id) for id in range(domain.shape[0]))
        total = np.array(total).reshape(-1,1) / neighbors

        # Greedily construct sample.
        for _ in range(size-1):

            # Find minimum distance to any point currently in sample.
            contrib = Parallel(n_jobs=-1)(delayed(compute_itrss)(domain[sample,:], id) for id in range(domain.shape[0]))
            contrib = np.array(contrib).reshape(-1,1) / (len(sample) + 1.0)
            score = total - contrib

            # Prevent previously chosen points from being chosen again.
            score[sample, :] = -999999

            # Sort points by minimum distance, find the max of this set.
            new_id = np.argsort(-score, axis=0)[0].item()

            # Add point to the sample.
            sample.append(new_id)

    else:

        # Find neighbors to use for later computations.
        nbrs = NearestNeighbors(n_neighbors=neighbors, metric='jaccard').fit(domain)
        _, indices = nbrs.kneighbors(domain)

        # Compute contributions to score from total dataset.
        def compute_itrss(input, id):
            datum = domain[id].reshape(1,-1)
            metric = tanimoto_kernel(input, datum)
            metric = np.sum(metric, axis=0)
            return metric
        total = Parallel(n_jobs=-1)(delayed(compute_itrss)(domain[indices[id]], id) for id in range(domain.shape[0]))
        total = np.array(total).reshape(-1,1) / neighbors

        # Greedily construct sample.
        for _ in range(size-1):

            # Find minimum distance to any point currently in sample.
            contrib = Parallel(n_jobs=-1)(delayed(compute_itrss)(domain[sample,:], id) for id in range(domain.shape[0]))
            contrib = np.array(contrib).reshape(-1,1) / (len(sample) + 1.0)
            score = total - contrib

            # Prevent previously chosen points from being chosen again.
            score[sample, :] = -999999

            # Find point with the maximum score.
            new_id = np.argsort(-score, axis=0)[0].item()

            # Add point to the sample.
            sample.append(new_id)

    # Return appropriate indices.
    if down_sample:
        return small_idx[sample].tolist()
    else:
        return sample

def vendi(domain, size, seed, metric='euclidean'):
    '''
        Select sample that maximizes the Vendi score for
        the chosen selection. The similarity function used
        in the Vendi score is the same probability
        distribution used for the max_entropy sample selection.
    '''

    # Redefine distances if dealing with Morgan fingerprints.
    if metric == 'jaccard':
        print('Sampling using jaccard distance!')

    # Set random seed.
    np.random.seed(seed=seed)

    # If the domain size is sufficiently small.
    down_sample = False
    if domain.shape[0] > 30000:
        down_sample = True
        print('Down-sampling...')
        domain_idx = [i for i in range(domain.shape[0])]
        small_idx = np.random.choice(domain_idx, size=10000, replace=False)
        domain = domain[small_idx]

    # Scale dataset.
    domain = MinMaxScaler().fit_transform(domain)

    # Choose the initial point.
    sample = [np.random.randint(low=0, high=domain.shape[0])]

    if metric == 'euclidean':

        # Get kernel bandwidths using Silverman's rule.
        stds = np.std(domain, axis=0)
        lower_quartile = np.percentile(domain, 25, axis=0)
        upper_quartile = np.percentile(domain, 75, axis=0)
        iqrs = upper_quartile - lower_quartile
        stack_ranges = np.vstack((stds, iqrs))
        metric = np.min(stack_ranges, axis=0)
        bandwidths = 0.9 * metric * domain.shape[0]**(-0.2)

        # For faster calculation, use a single bandwidth.
        optim_bandwidth = np.mean(bandwidths)

        # Calculation of Vendi score for a given matrix.
        def get_vendi_score(sample_indices):

            # Prepare covariance matrix.
            data_matrix = domain[sample_indices, :]
            distances = pdist(data_matrix)
            sim_matrix = squareform(distances)
            sim_matrix = np.exp(-np.square(sim_matrix) / (2.0 * optim_bandwidth**2))

            # Filter out extremely small values in the similarity matrix.
            sim_matrix = np.where(sim_matrix > 1e-20, sim_matrix, 0)
            sim_matrix = sim_matrix / sim_matrix.shape[0]

            # Compute Vendi score.
            try:
                eig_vals = np.linalg.eigvals(sim_matrix)
                eig_vals = np.abs(eig_vals)
                eig_vals = np.where(eig_vals > 1.0e-20, eig_vals, 1.0e-20)
                entropy = np.sum(np.multiply(eig_vals, np.log(eig_vals)))
                vendi_score = np.exp(-entropy)
                return vendi_score
            except:
                return 0.0
            
        # Helper function for computing the Vendi score for
        # a candidate point.
        def calc_vendi(index):
            test_sample = sample.copy()
            test_sample.append(index)
            return get_vendi_score(test_sample)
        
        # Greedily construct sample that maximizes the Vendi score.
        for _ in range(size-1):

            # In parallel, compute Vendi score with each data point added.
            vendi_scores = Parallel(n_jobs=-1)(delayed(calc_vendi)(id) for id in range(domain.shape[0]))
            vendi_scores = np.array(vendi_scores).reshape(-1,1)

            # Sort data by Vendi score.
            new_index = np.argsort(-vendi_scores, axis=0)[0].item()
            sample.append(new_index)

    else:

        # Calculation of Vendi score for a given matrix.
        def get_vendi_score(sample_indices):

            # Prepare covariance matrix.
            data_matrix = domain[sample_indices, :]
            distances = pdist(data_matrix, metric='jaccard')
            sim_matrix = squareform(distances)

            # Filter out extremely small values in the similarity matrix.
            sim_matrix = np.where(sim_matrix > 1e-20, sim_matrix, 0)

            # Compute Vendi score.
            try:
                eig_vals = np.linalg.eigvals(sim_matrix)
                eig_vals = np.abs(eig_vals)
                eig_vals = np.where(eig_vals > 1.0e-20, eig_vals, 1.0e-20)
                entropy = np.sum(np.multiply(eig_vals, np.log(eig_vals)))
                vendi_score = np.exp(-entropy)
                return vendi_score
            except:
                return 0.0
            
        # Helper function for computing the Vendi score for
        # a candidate point.
        def calc_vendi(index):
            test_sample = sample.copy()
            test_sample.append(index)
            return get_vendi_score(test_sample)
        
        # Greedily construct sample that maximizes the Vendi score.
        for _ in range(size-1):

            # In parallel, compute Vendi score with each data point added.
            vendi_scores = Parallel(n_jobs=-1)(delayed(calc_vendi)(id) for id in range(domain.shape[0]))
            vendi_scores = np.array(vendi_scores).reshape(-1,1)

            # Sort data by Vendi score.
            new_index = np.argsort(-vendi_scores, axis=0)[0].item()
            sample.append(new_index)

    # Return appropriate indices.
    if down_sample:
        return small_idx[sample].tolist()
    else:
        return sample

def tanimoto_kernel(X, Y):
    '''
        Helper method that computes the Tanimoto similarities between
        the rows of two numpy arrays.
    '''
    dot_product = np.dot(X, Y.T)
    norm_X = np.sum(X ** 2, axis=1)
    norm_Y = np.sum(Y ** 2, axis=1)
    kernel = dot_product / (norm_X[:, None] + norm_Y[None, :] - dot_product)
    return kernel

def sample(name, domain, size, seed, task='princeton'):
    '''
        Wrapper method for interfacing with the implemented
        sampling methods.
    '''
    if 'morgan' in task:
        dist_metric = 'jaccard'
    else:
        dist_metric = 'euclidean'

    if name == 'random':
        return random(domain=domain, size=size, seed=seed)
    elif name == 'maximin':
        return maximin(domain=domain, size=size, seed=seed, metric=dist_metric)
    elif name == 'medoids':
        return medoids(domain=domain, size=size, seed=seed, metric=dist_metric)
    elif name == 'max_entropy':
        return max_entropy(domain=domain, size=size, seed=seed, metric=dist_metric)
    elif name == 'vendi':
        return vendi(domain=domain, size=size, seed=seed, metric=dist_metric)
    else:
        raise Exception('The sampler name specified is not implemented.')

if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data, visualize

    # Load a dataset.
    dataset = load_data(task='princeton')

    # Select a sample.
    selected_indices = sample(name='max_entropy', domain=dataset[:,0:-1], size=10, seed=1)
    print(f'Selected indices: {selected_indices}')
    sample_points = dataset[selected_indices, :]

    # Visualize the sample.
    visualize(dataset=dataset, points=sample_points[:,0:-1])
