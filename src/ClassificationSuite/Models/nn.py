import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV
from skopt.callbacks import HollowIterationsStopper
from skopt.space import Real, Integer, Categorical

from ClassificationSuite.Models import AbstractModel

class NN(AbstractModel):
    '''
        Implementation of a two-layer neural network.
        Uncertainties are determined using an ensemble
        of neural networks; the number of MLPs used 
        can be set by the user.
    '''

    def __init__(self, n_models=10):
        super().__init__()
        self.name     = 'nn'
        self.model    = None
        self.n_models = n_models
        self.models   = []
        self.scaler   = None
        self.train_x  = None
        self.train_y  = None

    def train(self, cv=True, cv_score=False):
        '''
            Fit model to the loaded training data. This
            should involve hyperparameter tuning. This
            method should also deal with label changes
            or situations where only one class has been
            identified.
        '''

        # Check for exceptions.
        if not cv and cv_score:
            raise Exception('Cannot set cv=False and cv_score=True.')
        
        # Define scaler.
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_x)
        train_x = self.scaler.transform(self.train_x)
        
        # Determine number of folds for cross-validation.
        class_1 = np.where(self.train_y == 1)[0]
        class_2 = np.where(self.train_y == -1)[0]
        n_folds = np.min([len(class_1), len(class_2), 5])

        # If cross-validation is specified and we have enough folds.
        if cv and n_folds > 1:

            # Determine optimal hyperparameters given the training data.
            MLP_clf = BayesSearchCV(
                estimator=MLPWrapper(),
                search_spaces={
                    'layer1': Integer(4, 128),
                    'layer2': Integer(4, 128),
                    'activation': Categorical(['relu', 'tanh', 'logistic']),
                    'alpha': Real(1e-6, 1e-2, prior='log-uniform'),
                    'batch_size': Integer(4, 100),
                    'learning_rate': Categorical(['constant', 'adaptive']),
                    'learning_rate_init': Real(1e-5, 1e-1, prior='log-uniform'),
                },
                n_iter=100,
                scoring='f1_macro',
                n_jobs=1,
                n_points=1,
                cv=n_folds,
                refit=True,
                random_state=1,
                error_score=0.0
            )
            MLP_clf.fit(
                X=train_x, 
                y=self.train_y, 
                callback=HollowIterationsStopper(n_iterations=10, threshold=0.03)
            )
            self.model = MLP_clf.best_estimator_
            self.config = MLP_clf.best_params_

            # Train ensemble of neural networks using bootstrapping.
            self.models = []
            indices = [i for i in range(self.train_x.shape[0])]
            for _ in range(self.n_models):
                train_indices = np.random.choice(indices, size=int(0.7 * self.train_x.shape[0]))
                model = MLPWrapper(**self.config)
                model.fit(X=train_x[train_indices], y=self.train_y[train_indices])
                self.models.append(model)

            # Return CV score if specified.
            if cv_score:
                return np.max(MLP_clf.cv_results_['mean_test_score'])

        # If cross-validation is not specified or we don't have enough
        # folds.
        else:

            # If models and model haven't been instantiated.
            if len(self.models) == 0:
                for _ in range(self.n_models):
                    self.models.append(MLPWrapper())
            if self.model is None:
                self.model = MLPWrapper()

            # Train ensemble of models for uncertainty.
            indices = [i for i in range(self.train_x.shape[0])]
            for m in self.models:
                train_indices = np.random.choice(indices, size=int(0.7 * self.train_x.shape[0]))
                m.fit(train_x[train_indices], self.train_y[train_indices])

            # Train individual model on entire dataset.
            self.model.fit(X=train_x, y=self.train_y)

            # Report a CV score if requested.
            if cv_score:

                # If only one class, assume perfect prediction.
                if len(class_1) == 0 or len(class_2) == 0:
                    return 1.00
                
                # If one member of one class is found, assume it would
                # be predicted incorrectly for that fold in 5-fold CV.
                else:
                    return 0.8
    
    def classify(self, X_test, batch_size=50000):
        '''
            Provide classification labels for the provided
            data.
        '''
        if X_test.shape[0] < 100000:
            return self.model.predict(self.scaler.transform(X_test))
        else:
            factor = int(X_test.shape[0] / batch_size)
            n_batches = factor if X_test.shape[0] % batch_size == 0 else factor + 1
            y_all = []
            for batch_idx in range(n_batches):
                low = batch_idx * batch_size
                high = min((batch_idx + 1) * batch_size, X_test.shape[0])
                sample = X_test[low:high]
                y = self.model.predict(self.scaler.transform(sample))
                y_all.append(y)
            return np.hstack(y_all).reshape(-1)
    
    def predict(self, X_test, batch_size=50000):
        '''
            Provide classification probabilities for the provided data
            on a scale from -1 to 1.
        '''
        if X_test.shape[0] < 100000:
            y_pred_all = []
            for model in self.models:
                y_pred = model.predict(self.scaler.transform(X_test))
                y_pred_all.append(y_pred)
            return np.mean(y_pred_all, axis=0)
        else:
            factor = int(X_test.shape[0] / batch_size)
            n_batches = factor if X_test.shape[0] % batch_size == 0 else factor + 1
            y_all = []
            for batch_idx in range(n_batches):
                low = batch_idx * batch_size
                high = min((batch_idx + 1) * batch_size, X_test.shape[0])
                sample = X_test[low:high]
                y_pred = []
                for model in self.models:
                    y = model.predict(self.scaler.transform(sample))
                    y_pred.append(y)
                y_all.append(np.mean(y_pred, axis=0))
            return np.hstack(y_all).reshape(-1)
    
    def uncertainty(self, X_test, batch_size=50000):
        '''
            Provide uncertainty values for the provided
            data.
        '''
        if X_test.shape[0] < 100000:
            y_pred_all = []
            for model in self.models:
                y_pred = model.predict(self.scaler.transform(X_test))
                y_pred_all.append(y_pred)
            return np.std(y_pred_all, axis=0)
        else:
            factor = int(X_test.shape[0] / batch_size)
            n_batches = factor if X_test.shape[0] % batch_size == 0 else factor + 1
            y_all = []
            for batch_idx in range(n_batches):
                print(f'Uncertainty on batch {batch_idx+1} / {n_batches}.')
                low = batch_idx * batch_size
                high = min((batch_idx + 1) * batch_size, X_test.shape[0])
                sample = X_test[low:high]
                y_pred = []
                for model in self.models:
                    y = model.predict(self.scaler.transform(sample))
                    y_pred.append(y)
                y_all.append(np.std(y_pred, axis=0))
            return np.hstack(y_all).reshape(-1)
    
class MLPWrapper(BaseEstimator, ClassifierMixin):
    '''
        Wrapper for the sklearn MLPClassifier that allows
        for hyperparameter tuning of the number of neurons
        in each layer.
    '''
    def __init__(self, layer1=64, layer2=32, activation='relu', 
                 alpha=1e-4, batch_size='auto', learning_rate='constant', 
                 learning_rate_init=0.001, random_state=1):
        self.layer1 = layer1
        self.layer2 = layer2
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self.classes_ = [-1, 1]

    def fit(self, X, y):
        model = MLPClassifier(
            hidden_layer_sizes=[self.layer1, self.layer2],
            activation=self.activation,
            solver='lbfgs',
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=15000,
            random_state=self.random_state,
            early_stopping=True
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
    
if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output

    # Get a sample from a dataset.
    dataset = load_data(task='glotzer_pf')
    selected_indices = sample(name='medoids', domain=dataset[:,0:-1], size=50, seed=1)
    sample_points = dataset[selected_indices, :]

    # Train a model on the sample.
    model = MLP()
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