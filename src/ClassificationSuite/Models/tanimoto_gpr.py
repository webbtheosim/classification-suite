import gpytorch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import torch

from ClassificationSuite.Models import AbstractModel

class TanimotoGPR(AbstractModel):
    '''
        Implementation of a Gaussian process regression
        least-squares classifier that uses the acquisi-
        tion function outlined in Dai and Glotzer (2020).
    '''
    def __init__(self, ard=True):
        super().__init__()
        self.name    = 'gpr'
        self.ard     = ard
        self.scaler  = None
        self.train_x = None
        self.train_y = None
        print('Using Tanimoto GPR!')

    def train(self, cv=True, n_training_iter=10000, cv_score=False):
        '''
            Fit model to the loaded training data using
            the marginal log likelihood. If CV is not
            specified, than fixed points are adjusted
            without refitting kernel hyperparameters.
        '''

        # Throw an error if CV is False but CV-Score is True.
        if not cv and cv_score:
            raise Exception('Cannot specify cv=False and cv_score=True.')

        # If a CV score is needed.
        if cv_score:
            kf = KFold(n_splits=5)
            test_scores = []
            for index, (train_index, test_index) in enumerate(kf.split(self.train_x)):

                # Get train/test split for this fold.
                X_train = self.train_x[train_index]
                y_train = self.train_y[train_index]
                X_test = self.train_x[test_index]
                y_test = self.train_y[test_index]

                # Convert training data to appropriate format. Keep scalar
                # for prediction tasks too.
                sc = MinMaxScaler()
                sc.fit(X_train)
                y_train = np.where(y_train == -1, 0, 1)
                train_x_processed = torch.DoubleTensor(sc.transform(X_train))
                train_y_processed = torch.DoubleTensor(y_train)

                # Define model.
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = ExactGPModel(
                    train_x = train_x_processed,
                    train_y = train_y_processed,
                    likelihood = likelihood,
                    ard = self.ard
                )
                model.double()

                # Set up training.
                model.train()
                likelihood.train()
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

                # Train model.
                training_iterations = 1000
                loss_prev = 0.0
                for i in range(training_iterations):
                    optimizer.zero_grad()
                    output = model(train_x_processed)
                    loss = -mll(output, train_y_processed)
                    loss.backward()
                    if abs(loss.item() - loss_prev) <= 1e-6:
                        break
                    optimizer.step()
                
                # Get predictions on test data.
                X_test_processed = torch.DoubleTensor(sc.transform(X_test))
                model.eval()
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    output = likelihood(model(X_test_processed))
                    mean = output.mean.detach().numpy()
                    y_pred = np.where(mean > 0, 1, -1)

                # Get test score for this fold.
                test_scores.append(f1_score(y_test, y_pred, average='macro'))

        # Reset fixed points and refit hyperparameters.
        if cv:

            # Convert training data to appropriate format. Keep scalar
            # for prediction tasks too.
            sc = MinMaxScaler()
            sc.fit(self.train_x)
            self.scaler = sc
            train_x_processed = torch.DoubleTensor(sc.transform(self.train_x))
            train_y_processed = torch.DoubleTensor(self.train_y)

            # Define a new model with the appropriate information.
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = ExactGPModel(
                train_x = train_x_processed,
                train_y = train_y_processed,
                likelihood = self.likelihood,
                ard = self.ard
            )
            self.model.double()

            # Fit hyperparameters to training data.
            self.model.train()
            self.likelihood.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
            loss_prev = 0
            for i in range(n_training_iter):
                optimizer.zero_grad()
                output = self.model(train_x_processed)
                loss = -mll(output, train_y_processed)
                loss.backward()
                if abs(loss.item() - loss_prev) <= 1e-6:
                    break
                loss_prev = loss.item()
                optimizer.step()

        # Set fixed points upon which GP bases its decisions.
        else:

            # Convert training data to appropriate format. Keep scalar
            # for prediction tasks too.
            sc = MinMaxScaler()
            sc.fit(self.train_x)
            self.scaler = sc
            train_x_processed = torch.DoubleTensor(sc.transform(self.train_x))
            train_y_processed = torch.DoubleTensor(self.train_y)

            # Reset fixed points.
            self.model.set_train_data(
                inputs=train_x_processed, 
                targets=train_y_processed,
                strict=False
            )

        # Return cv_score if quested.
        if cv_score:
            print(test_scores)
            return np.mean(test_scores)
    
    def classify(self, X_test, logits=False):
        '''
            Provide classification labels for the provided data.
        '''
        # Scale input.
        input_processed = torch.DoubleTensor(self.scaler.transform(X_test))

        # Make predictions.
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.likelihood(self.model(input_processed))
            mean = output.mean.detach().numpy()
            if logits:
                return mean
            else:
                return np.where(mean > 0, 1, -1)
            
    def predict(self, X_test):
        '''
            Provide classification probabilities for the provided data
            on a scale from -1 to 1.
        '''
        # Scale input.
        input_processed = torch.DoubleTensor(self.scaler.transform(X_test))

        # Make predictions.
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.likelihood(self.model(input_processed))
            mean = output.mean.detach().numpy()
            return mean
    
    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided data.
            Here, the uncertainty is based on the acquisition
            function specified by Dai and Glotzer (2020).
        '''
        # Scale input.
        input_processed = torch.DoubleTensor(self.scaler.transform(X_test))

        # Make predictions.
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.likelihood(self.model(input_processed))
            y_std = output.stddev.numpy()
        y_pred = self.classify(X_test=X_test, logits=True)
        y_acq = y_std / (np.abs(y_pred) + 0.05)

        return y_acq
    
class ExactGPModel(gpytorch.models.ExactGP):
    '''
        Defines ExactGPModel according to GPytorch documentation.
        Includes both isotropic and anistropic RBF kernels.
    '''
        
    def __init__(self, train_x, train_y, likelihood, ard=True):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class TanimotoKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    has_lengthscale = False
    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)
    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        return batch_tanimoto_sim(x1, x2)
    
def batch_tanimoto_sim(x1, x2):
    assert x1.ndim >= 2 and x2.ndim >= 2
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_sum = torch.sum(x1 ** 2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2 ** 2, dim=-1, keepdims=True)
    return (dot_prod) / (x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod)
    
if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output

    # Get a sample from a dataset.
    dataset = load_data(task='qm9_gap_morgan')
    selected_indices = sample(name='random', domain=dataset[:,0:-1], size=50, seed=1)
    sample_points = dataset[selected_indices, :]

    # Train a model on the sample.
    model = TanimotoGPR()
    model.load_data(X=sample_points[:,0:-1], y=sample_points[:,-1])
    cv_score = model.train(cv=True, cv_score=True)
    print(f'CV score: {cv_score}')
    y_pred = model.classify(X_test=dataset[:,0:-1], logits=False)
    y_acq = model.uncertainty(X_test=dataset[:,0:-1])

    # Get scores.
    print(f'Scores: {model.score(X_test=dataset[:,0:-1], y_test=dataset[:,-1])}')

    # # Visualize predictions of model.
    # visualize_model_output(
    #     dataset=dataset, 
    #     chosen_points=sample_points[:,0:-1], 
    #     y_pred=y_pred,
    #     y_acq=y_acq
    # )