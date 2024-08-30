import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as td
import torchbnn as bnn
from typing import OrderedDict

from ClassificationSuite.Models import AbstractModel

class BKDE(AbstractModel):
    '''
        An adaptation of the Gryffin algorithm originally
        introduced by Hase et al. (2020). Bayesian kernel
        density estimates for each class are used to compute
        labels, and their normalized values are treated as
        probabilities for an entropy-based uncertainty.

        To reduce computational time, Gryffin has a custom
        batch selection scheme based on its kernel density
        estimates.

        Note: Similar to label propagation, the surrogate
        model is constructed when asked to classify. Un-
        certainties should always be calculated after call-
        ing 'classify' and should have the same domain.
    '''

    def __init__(self):
        super().__init__()
        self.name    = 'bkde'
        self.scaler  = None
        self.train_x = None
        self.train_y = None

    def train(self, cv=False):
        '''
            Fit model to the loaded training data. This
            should involve hyperparameter tuning. This
            method should also deal with label changes
            or situations where only one class has been
            identified.
        '''
        
        # Scale the training data.
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_x)
        train_x = self.scaler.transform(self.train_x)

        # Define and fit a Bayesian autoencoder.
        self.bae = BayesianAutoencoder(
            in_dim = train_x.shape[1],
            n_obs = train_x.shape[0]
        )
        self.bae.fit(train_x=train_x)

        # Build kernels using the trained Bayesian autoencoder.
        self.kernels = self.bae.compute_kernels(X=train_x, y=self.train_y)

    def classify(self, X_test):
        '''
            Provide classification labels for the provided
            data. The surrogate model is built in this method.
            
            Gryffin is constructed so that the prior bandwidth distribution
            that narrows as training data is acquired. This leads to bandwidths
            becoming so narrow that kernels tend to zero at some points in the
            domain, resulting in zero values. However, in this limit, the label
            of a point in the domain should just be the same as the label of the
            closest labelled point. To implement this, we use the NearestNeighbors
            method from sklearn.
        '''

        # Scale the domain.
        X_sc = self.scaler.transform(X_test)

        # Build the surrogate model on this domain.
        self._build_surrogate(X_sc)

        # Get prediction. 
        y_pred = np.where(self.surrogate[:,0] > self.surrogate[:,1], 1, -1)

        # Deal with domain regions where the surrogate is sparse.
        sparse_indices = np.where(self.surrogate[:,0] == self.surrogate[:,1])[0]
        if len(sparse_indices) > 0:
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X=self.scaler.transform(self.train_x), y=self.train_y)
            y_pred[sparse_indices] = knn.predict(X_sc[sparse_indices])

        return y_pred
    
    def _build_surrogate(self, X):
        '''
            Helper method for applying the kernels to the domain
            specified by X.
        '''

        # Set precision to lower memory requirements.
        PRECISION = np.float32
        X = X.astype(PRECISION)

        # Get indices for data associated with each class.
        indices_1 = np.where(self.kernels['label'] == 1)[0]
        indices_0 = np.where(self.kernels['label'] == -1)[0]

        # Build surrogate models for each label.
        surrogate_1 = np.zeros((X.shape[0])).astype(PRECISION)
        surrogate_0 = np.zeros((X.shape[0])).astype(PRECISION)
        n_samples = self.kernels['loc'].shape[0]
        for sample_index in range(n_samples):

            # Get locs, scales for this sample.
            locs_1 = self.kernels['loc'][sample_index, indices_1, :].astype(PRECISION)
            scales_1 = self.kernels['scale'][sample_index, indices_1, :].astype(PRECISION)
            locs_0 = self.kernels['loc'][sample_index, indices_0, :].astype(PRECISION)
            scales_0 = self.kernels['scale'][sample_index, indices_0, :].astype(PRECISION)

            # Compute positive contributions on the domain.
            d2_1 = -np.sum((X[:, np.newaxis, :] - locs_1[np.newaxis, :, :])**2 / (2 * scales_1[np.newaxis, :, :]**2), axis=-1)
            gauss_1 = np.exp(d2_1) / np.sqrt(np.power(2.0 * np.pi, int(X.shape[1])) * np.prod(np.square(scales_1), axis=1))
            del d2_1
            all_1 = np.sum(gauss_1, axis=1)
            del gauss_1
            surrogate_1 = surrogate_1 + all_1
            del all_1

            # Compute negative contributions on the domain.
            d2_0 = -np.sum((X[:, np.newaxis, :] - locs_0[np.newaxis, :, :])**2 / (2 * scales_0[np.newaxis, :, :]**2), axis=-1)
            gauss_0 = np.exp(d2_0) / np.sqrt(np.power(2.0 * np.pi, int(X.shape[1])) * np.prod(np.square(scales_0), axis=1))
            del d2_0
            all_0 = np.sum(gauss_0, axis=1)
            del gauss_0
            surrogate_0 = surrogate_0 + all_0
            del all_0

        # Take averages, per Hase et al. (2020).
        surrogate_1 /= n_samples
        surrogate_0 /= n_samples

        # Save surrogate as a numpy array of shape (X.shape[0], 2).
        self.surrogate = np.hstack((surrogate_1.reshape(-1,1), surrogate_0.reshape(-1,1)))

    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided
            data. This requires having the surrogate model build
            in the previous method.
        '''

        # For regions where kernel density estimates are sparse
        # for both labels, their distribution ([0.0, 0.0]) should
        # be considered uniform ([0.5, 0.5]).
        surrogate = self.surrogate.copy()
        for row in range(self.surrogate.shape[0]):
            if self.surrogate[row,0] == self.surrogate[row,1]:
                surrogate[row,:] = [0.5, 0.5]

        return entropy(surrogate, axis=1)
    
    def recommend(self, domain, size, chosen_indices):
        '''
            Recommend 'size' number of points to be measured next from 
            the available 'domain.' Returns the indices associated with 
            the domain points chosen. This method chooses a batch using
            local penalization based on the kernel density estimate of
            the point that's been chosen.
        '''
        sample = []
        y_acq = self.uncertainty(X_test=domain)
        y_acq[chosen_indices] = -999999
        for _ in range(size):

            # Choose a point that maximizes the acquisition function.
            new_index = np.argmax(y_acq)
            sample.append(new_index)

            # Check to see if we're finished.
            if len(sample) == size:
                break

            # Get kernel density estimate for the selected point.
            X_new = domain[new_index].reshape(1,-1)
            loc_new, scale_new = self.bae.get_kernel_density(X=X_new)
            n_samples = loc_new.shape[0]

            # Penalize acquisition function by that Gaussian.
            X_copied = np.repeat(domain[np.newaxis, :, :], n_samples, axis=0)
            d2 = -np.sum((X_copied - loc_new)**2 / (2 * scale_new**2), axis=-1)
            # gauss = np.exp(d2) / np.prod(np.sqrt(2 * np.pi) * scale_new, axis=-1)
            gauss = np.exp(d2)
            gauss = np.mean(gauss, axis=0)
            penalties = 1.0 - gauss.reshape(-1)
            y_acq = y_acq * penalties

            # Prevent the same point from being chosen.
            y_acq[sample] = -999999

        return sample
    
class BayesianAutoencoder(nn.Module):
    '''
        Implementation of a Bayesian autoencoder used for kernel
        density estimation. The architecture is based on the 
        architecture typically adopted by Phoenics/Gryffin, i.e. 
        three hidden layers of six nodes each.

        Unlike the current implementation of Gryffin, we found
        improved fitting performance by setting prior sigmas to 0.1
        instead of 1.0.
    '''
    def __init__(self, in_dim, n_obs, prior_mu=0.0, prior_sigma=0.1,
                 hidden_dim=24, n_hidden=3, n_draws=100):
        super().__init__()
        self.in_dim  = in_dim
        self.n_obs   = n_obs
        self.n_draws = n_draws

        # Build autoencoder architecture.
        layers = [
            ('linear_0', bnn.BayesLinear(
                prior_mu=prior_mu, 
                prior_sigma=prior_sigma,
                in_features=in_dim,
                out_features=hidden_dim,
                bias=True
            )),
            ('relu_0', nn.ReLU())
        ]
        for idx in range(1,n_hidden):
            layers.append(
                (f'linear_{idx}', bnn.BayesLinear(
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    bias=True
                ))
            )
            layers.append(
                (f'relu_{idx}', nn.ReLU())
            )
        layers.append(
            (f'linear_{n_hidden}', bnn.BayesLinear(
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    in_features=hidden_dim,
                    out_features=in_dim,
                    bias=True
            ))
        )
        self.layers = nn.Sequential(OrderedDict(layers))

        # Instantiate parameters for kernel bandwidths.
        self.gamma_concentration = nn.Parameter(
            torch.zeros(n_obs, in_dim) + 12*(n_obs**2)
        )
        self.gamma_rate = nn.Parameter(torch.ones(n_obs, in_dim))
        self.tau = td.gamma.Gamma(
            F.softplus(self.gamma_concentration, threshold=0.01),
            F.softplus(self.gamma_rate, threshold=0.01)
        )

        # Set model to float type.
        self.float()

        # Define numpy graph used for sampling and kernel computations.
        self.numpy_graph = NumpyGraph(n_obs, in_dim, n_hidden, hidden_dim, n_draws)

    def forward(self, X, y):

        # If not, use the full set of values.
        X = self.layers(X)
        scale = 1.0 / torch.sqrt(td.gamma.Gamma(
            F.softplus(self.gamma_concentration, threshold=0.01),
            F.softplus(self.gamma_rate, threshold=0.01)
        ).rsample())

        # Prepare output, per Gryffin implementation.
        X = 1.2 * torch.sigmoid(X) - 0.1
        pred = td.normal.Normal(X, scale)
        out = {'pred': pred, 'target': y}

        return out
    
    def fit(self, train_x, max_epochs=100000, lr=1e-3, verbose=True):
        '''
            Method for training the Bayesian autoencoder on the
            provided data.
        '''

        # Convert training data to tensor format.
        train_x = torch.FloatTensor(train_x)

        # Initial training loop.
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', 
            factor=0.5,
            patience=1000
        )
        for t in range(max_epochs):

            # Training.
            self.train()
            inference = self.forward(X=train_x, y=train_x)
            loss = -torch.sum(inference['pred'].log_prob(inference['target']))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            # Check for convergence.
            if optimizer.param_groups[0]['lr'] < 1e-5:
                print(f'Stopping early at {t+1} epochs!')
                break

            # Report results.
            if verbose and (t+1) % 1000 == 0:
                print(f'Epoch: {t+1} | Loss: {loss.item()}')

    def sample(self):
        '''
            Method for sampling parameters from the BNN. These samples
            are used by the compute_kernels method.
        '''

        # Sample from the Bayesian autoencoder.
        samples = {}
        idx = 0
        for _, module in self.layers.named_modules():
            if isinstance(module, bnn.BayesLinear):
                weight_dist = td.normal.Normal(module.weight_mu, torch.exp(module.weight_log_sigma))
                bias_dist = td.normal.Normal(module.bias_mu, torch.exp(module.bias_log_sigma))
                weight_sample = weight_dist.sample(sample_shape=(self.n_draws,1)).squeeze()
                bias_sample = bias_dist.sample(sample_shape=(self.n_draws,1)).squeeze()
                weight_sample = weight_sample.transpose(-2,-1)
                samples[f'weight_{idx}'] = weight_sample.numpy()
                samples[f'bias_{idx}'] = bias_sample.numpy()
                idx += 1

        # Sample from the gamma distribution.
        # Note: This is a deviation from what is done in the Gryffin implementation.
        # For some reason, they don't create a new Gamma distribution like they do for
        # the modules above; this returns just the prior parameters, given what I've seen.
        # The implementation here samples from the fitted parameters, as is intended by 
        # the original manuscript.
        gamma_dist = td.gamma.Gamma(
            F.softplus(self.gamma_concentration, threshold=0.01), 
            F.softplus(self.gamma_rate, threshold=0.01)
        )
        samples['gamma'] = gamma_dist.sample(sample_shape=(self.n_draws,1)).squeeze().numpy()

        # Save samples.
        self.samples = samples

    def compute_kernels(self, X, y):
        '''
            Wrapper method for sampling from the Bayesian autoencoder,
            computing kernels with the NumpyGraph, and saving these
            kernels in the desired format.
        '''

        # Sample from the Bayesian autoencoder.
        self.sample()

        # Use numpy graph to evaluate kernels.
        self.numpy_graph.load(X=X)
        self.kernels = self.numpy_graph.compute_kernels(self.samples)

        # Add labels to the kernel information.
        self.kernels['label'] = y

        return self.kernels
    
    def get_kernel_density(self, X):
        '''
            Method used by the Gryffin 'recommend' method to get the loc
            and scale of a kernel density estimate for a specified set
            of points.
        '''
        self.numpy_graph.load(X=X)
        new_kernels = self.numpy_graph.compute_kernels(self.samples)
        locs   = new_kernels['loc']

        # Since we haven't fit to this data point, we can determine scales
        # from the prior distribution.
        taus = np.random.gamma((12 * self.n_obs**2) + np.zeros(X.shape), np.ones(X.shape))
        scales = 1.0 / np.sqrt(taus)

        return (locs, scales)

class NumpyGraph:
    '''
        Helper class used to efficiently sample from a Bayesian
        autoencoder for kernel computations.
    '''
    def __init__(self, n_obs, in_dim, n_hidden, hidden_dim, n_draws=100):
        self.n_obs      = n_obs
        self.in_dim     = in_dim
        self.n_hidden   = n_hidden
        self.hidden_dim = hidden_dim
        self.n_draws    = n_draws

    def sigmoid(self, X):
        return 1. / (1. + np.exp(-X))
    
    def load(self, X):
        '''Loads input data for sampling from the BNN.'''
        self.n_obs = len(X)
        self.X     = X

    def compute_kernels(self, samples):
        '''
            This method takes as input samples from the Bayesian
            autoencoder. These samples are used to compute the 
            'locs' and 'scales' for the kernels used to make 
            predictions. The results are sotred in a dictionary
            with attributes 'loc', 'sqrt_prec', and 'scale'.
        '''

        # Build Bayesian autoencoder architecture with numpy, and draw
        # samples using this numpy architecture.
        activations = [lambda x: np.maximum (x,0) for i in range(self.n_hidden+1)]
        activations.append(lambda x: x)
        layer_outputs = [np.array([self.X for _ in range(self.n_draws)])]
        for layer_index in range(self.n_hidden+1):
            weight = samples[f'weight_{layer_index}']
            bias = samples[f'bias_{layer_index}']
            activation = activations[layer_index]
            outputs = []
            for sample_index in range(len(weight)):
                single_weight = weight[sample_index]
                single_bias = bias[sample_index]
                output = activation(np.matmul(layer_outputs[-1][sample_index], single_weight) + single_bias)
                outputs.append(output)
            layer_output = np.array(outputs)
            layer_outputs.append(layer_output)
        bnn_output = layer_outputs[-1]

        # Compute kernels using the outputs of the numpy Bayesian autoencoder.
        # NOTE: In the Gryffin implementation on Github, the following line actually
        # draws values from a gamma distribution with prior parameters. I don't 
        # know why they do this, considering that gamma parameters are part of 
        # what is fit by the network. We use the fitted gamma parameters here.
        tau = samples['gamma']
        sqrt_tau = np.sqrt(tau)
        scale = 1. / sqrt_tau
        loc = 1.2 * (self.sigmoid(bnn_output) - 0.1)

        kernels = {
            'loc': loc,
            'sqrt_prec': sqrt_tau,
            'scale': scale
        }

        return kernels

if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output
    torch.manual_seed(1)

    # Get a sample from a dataset.
    dataset = load_data(task='princeton')
    selected_indices = sample(name='random', domain=dataset[:,0:-1], size=10, seed=1)
    sample_points = dataset[selected_indices,:]

    # Train a model on the sample.
    model = Gryffin()
    model.load_data(X=sample_points[:,0:-1], y=sample_points[:,-1])
    model.train()
    y_pred = model.classify(X_test=dataset[:,0:-1])
    y_acq = model.uncertainty(X_test=dataset[:,0:-1])

    # Get scores.
    print(f'Scores: {model.score(X_test=dataset[:,0:-1], y_test=dataset[:,-1])}')

    # Visualize predictions of model.
    visualize_model_output(
        dataset=dataset, 
        chosen_points=sample_points[:,0:-1], 
        y_pred=y_pred,
        y_acq=y_acq
    )