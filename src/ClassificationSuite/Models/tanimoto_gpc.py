import gpytorch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import torch

from ClassificationSuite.Models import AbstractModel

class TanimotoGPC(AbstractModel):
    '''
        Implementation of a Gaussian process classifier
        that uses a Bernoulli likelihood.
    '''

    def __init__(self, ard=True):
        super().__init__()
        self.name    = 'gpc'
        self.ard     = ard
        self.scaler  = None
        self.train_x = None
        self.train_y = None
        print('Using Tanimoto GPC!')

    def train(self, cv=False, cv_score=False):
        '''
            Fit model to the loaded training data using
            the marginal log likelihood. CV is ignored
            here because fixed points cannot be adjusted 
            without tuning hyperparameters, per gpytorch.
        '''

        # Throw error is CV is False but CV-Score is True.
        if not cv and cv_score:
            raise Exception('Cannot specify cv=False and cv_score=True.')
        
        # If a cross-validation score is needed.
        if cv_score:

            # Perform 5-fold CV.
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
                model = ClassificationGP(train_x_processed, train_y_processed, ard=self.ard)
                likelihood = gpytorch.likelihoods.BernoulliLikelihood()
                model.double()

                # Set up training.
                model.train()
                likelihood.train()
                mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y_processed.numel())
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
                    classes = output.mean.float().numpy()
                    classes = 1.0 * classes + -1.0 * (1.0 - classes)
                    y_pred = np.where(classes > 0.0, 1, -1)

                # Get test score for this fold.
                test_scores.append(f1_score(y_test, y_pred, average='macro'))

        # Convert training data to appropriate format. Keep scalar
        # for prediction tasks too.
        sc = MinMaxScaler()
        sc.fit(self.train_x)
        self.scaler = sc
        train_y = np.where(self.train_y == -1, 0, 1)
        train_x_processed = torch.DoubleTensor(sc.transform(self.train_x))
        train_y_processed = torch.DoubleTensor(train_y)

        # Define model.
        model = ClassificationGP(train_x_processed, train_y_processed, ard=self.ard)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        model.double()

        # Set up training.
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y_processed.numel())
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
        
        # Save model and parameters.
        self.model = model
        self.likelihood = likelihood

        # Return CV score if requested.
        if cv_score:
            return np.mean(test_scores)

    def classify(self, X_test):
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
            classes = output.mean.float().numpy()
            classes = 1.0 * classes + -1.0 * (1.0 - classes)
            return np.where(classes > 0.0, 1, -1)
        
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
            classes = output.mean.float().numpy()
            classes = 1.0 * classes + -1.0 * (1.0 - classes)
            return classes
        
    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided data.
        '''
        # Scale input.
        input_processed = torch.DoubleTensor(self.scaler.transform(X_test))

        # Make predictions.
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.likelihood(self.model(input_processed))
            y_std = output.stddev.numpy()
            return y_std

class ClassificationGP(gpytorch.models.ApproximateGP):

    def __init__(self, train_x, train_y, ard=True):
        self.train_y = train_y
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(ClassificationGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
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
    model = TanimotoGPC(ard=True)
    model.load_data(X=sample_points[:,0:-1], y=sample_points[:,-1])
    cv_score = model.train(cv=True, cv_score=True)
    print(f'CV Score: {cv_score}')
    y_pred = model.classify(X_test=dataset[:,0:-1])
    y_acq = model.uncertainty(X_test=dataset[:,0:-1])

    # Get scores.
    print(f'Scores: {model.score(X_test=dataset[:,0:-1], y_test=dataset[:,-1])}')