from itertools import chain, combinations
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from ClassificationSuite.Models import AbstractModel
from ClassificationSuite.Models.utils import get_model

class AbstractEnsemble(AbstractModel):

    def __init__(self, models):
        super().__init__()
        self.name      = 'Abstract Ensemble'
        self.top_model = None

        # Check for exceptions.
        if 'lp' in models or 'bkde' in models:
            raise Exception('Models cannot include LP or BKDE.')

        # Get models from the model names provided.
        self.models = []
        for m in models:
            self.models.append(get_model(name=m))

    def load_data(self, X, y):
        '''
            Save training data at the class level.
            If called repeatedly, the training data
            will be overridden.
        '''
        # Store data in class.
        self.train_x = X
        self.train_y = y.reshape(-1)

        # Load data to models.
        for m in self.models:
            m.load_data(X=X, y=y)

    def add_data(self, X_new, y_new):
        '''
            Add new training data to an existing set.
        '''
        # Store data in class.
        if self.train_x is None:
            self.train_x = X_new
        else:
            self.train_x = np.vstack((
                self.train_x, X_new.reshape(-1, self.train_x.shape[1])
            ))
        if self.train_y is None:
            self.train_y = y_new
        else:
            self.train_y = np.hstack((
                self.train_y, y_new.reshape(-1)
            ))

        # Load data to models.
        for m in self.models:
            m.add_data(X_new=X_new, y_new=y_new)

class TopModelEnsemble(AbstractEnsemble):
    '''
        Implementation of an ensemble model that relies only on the
        predictions of the invidual model with the highest cross-
        validation score upon training.
    '''

    def __init__(self, models):
        super().__init__(models=models)
        self.name      = 'ensemble_top'
        self.top_model = None

    def train(self, cv=False):
        '''
            Each individual model will be trained with hyperparameter
            tuning on the loaded data. The model with the highest cross-
            validation score will be stored for future work. If cv is
            False, only the top performing model is retrained.
        '''

        # If cross-validation is specified, train all of the models.
        if cv:
            
            # Record CV scores of all models.
            cv_scores = [0.0 for _ in range(len(self.models))]
            for index, m in enumerate(self.models):
                # print(f'Training model {m.name}.')
                cv_scores[index] = m.train(cv=True, cv_score=True)

            # Determine the model with the top score.
            top_index = np.argmax(cv_scores)
            self.top_model = self.models[top_index]
            # for index, score in enumerate(cv_scores):
            #     print(self.models[index].name, score)
            # print(f'Chosen model is: {self.top_model.name}')
            
        # If cross-validation is not specified, just refit the top
        # model without hyperparameter tuning.
        else:
            self.top_model.train(cv=False)
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided data. For this
            ensemble scheme, only the top model is used for prediction.
        '''
        return self.top_model.classify(X_test)
    
    def uncertainty(self, X_test):
        '''
            Provide classification labels for the provided data. For this
            ensemble scheme, only the top model is used for uncertainty.
        '''
        return self.top_model.uncertainty(X_test)
    
    def recommend(self, domain, size, chosen_indices):
        '''
            Recommend 'size' number of points to be
            measured next from the available 'domain.'
            Returns the indices associated with the
            domain points chosen. This method uses the 
            kriging believer scheme, where models are re-
            trained without CV.
        '''

        sample = []
        for _ in range(size):

            # Compute acquisition function.
            y_pred = self.top_model.classify(X_test=domain)
            y_acq = self.top_model.uncertainty(X_test=domain)
            y_acq[sample] = -999999
            y_acq[chosen_indices] = -999999
            
            # Add jitter values to break ties.
            y_acq = y_acq + 1e-50 * np.random.random(size=y_acq.shape[0])

            # Choose point that maximizes uncertainty.
            new_index = np.argsort(-y_acq)[0].item()
            sample.append(new_index)

            # Add fake points to the training data.
            new_feature = domain[new_index, :].reshape(1,-1)
            new_label = y_pred[new_index].reshape(-1)
            self.add_data(X_new=new_feature, y_new=new_label)

            # Retrain model.
            self.top_model.train(cv=False)

        return sample

class AveragingEnsemble(AbstractEnsemble):
    '''
        Implementation of an ensemble model that considers every subset
        of models provided. Model predictions and uncertainties are aver-
        aged together. The combination of models with the highest cross-
        validation score is used for prediction and acquisition.
    '''

    def __init__(self, models):
        super().__init__(models=models)
        self.name            = 'ensemble_averaging'
        self.selected_models = None

    def train(self, cv=False):
        '''
            Each individual model will be trained with hyperparameter
            tuning on the loaded data. Models will be kept in the ensemble
            so long as they improve prediction performance on the valid-
            ation set.
        '''

        # If cross-validation is specified, train all of the models.
        if cv:
            
            # Construct one held-out validation set.
            X_train, X_test, y_train, y_test = train_test_split(
                self.train_x, self.train_y, test_size=0.2)

            # Train all models on the training set.
            y_preds = np.zeros((len(self.models), y_test.shape[0]))
            for index, m in enumerate(self.models):
                m.load_data(X=X_train, y=y_train)
                m.train(cv=True, cv_score=False)
                y_preds[index, :] = m.predict(X_test=X_test)

            # Consider every combination of models.
            powerset = list(self._powerset([i for i in range(len(self.models))]))[1:]
            cv_scores = []
            for index, indices in enumerate(powerset):
                y_pred = np.mean(y_preds[indices,:], axis=0)
                y_pred = np.where(y_pred > 0, 1, -1)
                cv_scores.append(
                    f1_score(
                        y_true=y_test, 
                        y_pred=y_pred, 
                        average='macro'
                    )
                )
            top_subset = powerset[np.argmax(cv_scores)]
            # for index, score in enumerate(cv_scores):
            #     print(powerset[index], score)

            # Save the top subset of models.
            self.selected_models = []
            for index, m in enumerate(self.models):
                if index in top_subset:
                    self.selected_models.append(m)
            print(f'Selected models: {[m.name for m in self.selected_models]}')

            # Retrain models on all training data.
            for m in self.models:
                m.load_data(X=self.train_x, y=self.train_y)
                m.train(cv=True)
            
        # If cross-validation is not specified, just refit the selected
        # models without hyperparameter tuning.
        else:
            for m in self.selected_models:
                m.train(cv=False)

    def _powerset(self, iterable):
        ''' Helper method that generates every subset of a list. '''
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided data. For this
            ensemble scheme, we take an average over selected models.
        '''
        y_pred_all = []
        for model in self.selected_models:
            y_pred = model.predict(X_test)
            y_pred_all.append(y_pred)
        y_pred = np.mean(y_pred_all, axis=0)
        return np.where(y_pred > 0, 1, -1)
    
    def uncertainty(self, X_test):
        '''
            Provide classification labels for the provided data. For this 
            ensemble scheme, we take an average over selected models.
        '''

        # Get uncertainties from all models.
        y_var_all = []
        for model in self.selected_models:
            y_var = model.uncertainty(X_test)
            y_var_all.append(y_var)

        # Put the uncertainties from each model on the same scale.
        y_var_all = np.transpose(y_var_all)
        y_var_all /= np.max(y_var_all, axis=0)

        return np.mean(y_var_all, axis=1)

class StackingEnsemble(AbstractEnsemble):
    '''
        Implementation of an ensemble model that weighs the pred-
        ictions and uncertainties of each component. Weights are
        determined using a non-negative linear regression model, 
        and weights are averaged over each fold in cross validation.
    '''

    def __init__(self, models, stack_model='weights'):
        super().__init__(models=models)
        self.name        = 'ensemble_stacking'
        self.stack_model = stack_model
        self.weights     = np.array([1.0 / len(self.models) for _ in self.models])

    def train(self, cv=False):
        '''
            Each individual model will be trained with hyperparameter
            tuning on the loaded data. Models will be kept in the ensemble
            so long as they improve prediction performance on the valid-
            ation set.
        '''

        # If cross-validation is specified, train all of the models.
        if cv:

            # Determine the optimal hyperparameters for this
            # training set.
            for m in self.models:
                m.train(cv=True)

            # Only run CV if there is sufficient coverage for both classes.
            # If not, just use even weights, as initialized above.
            class_1 = np.where(self.train_y == 1)[0]
            class_2 = np.where(self.train_y == -1)[0]
            n_folds = np.min([len(class_1), len(class_2), 5])
            if n_folds > 1:
                all_weights = []
                kf = KFold(n_splits=n_folds)
                for index, (train_index, test_index) in enumerate(kf.split(self.train_x)):

                    # Get train/val split.
                    X_train = self.train_x[train_index]
                    y_train = self.train_y[train_index]
                    X_test = self.train_x[test_index]
                    y_test = self.train_y[test_index]

                    # Load current fold to models.
                    y_preds = np.zeros((len(self.models), y_test.shape[0]))
                    for i, m in enumerate(self.models):
                        m.load_data(X=X_train, y=y_train)
                        m.train(cv=False)
                        y_preds[i, :] = m.predict(X_test=X_test)
                    y_preds = np.transpose(y_preds)
                    
                    # Fit stack model.
                    if self.stack_model == 'weights':
                        try:
                            weights, _ = nnls(A=y_preds, b=y_test, maxiter=int(30 * y_preds.shape[1]))
                            all_weights.append(weights)

                        # If there's problems with nnls fitting, just recommend
                        # whatever self.weights currently is.
                        except:
                            all_weights.append(self.weights.tolist())

                # Get normalized weights.
                weights = np.sum(all_weights, axis=0)
                self.weights = weights / np.sum(weights)
                # print(f'Weights: {self.weights}')

                # Retrain models on entire dataset. Hyperparameters are
                # already determined.
                for m in self.models:
                    m.load_data(X=self.train_x, y=self.train_y)
                    m.train(cv=False)

        # If CV is not specified, just train each component.
        else:
            for m in self.models:
                m.train(cv=False)
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided data. For this
            ensemble scheme, we take a weighted average over predictions.
        '''
        y_pred_all = []
        for model in self.models:
            y_pred = model.predict(X_test)
            y_pred_all.append(y_pred)
        y_pred = np.matmul(self.weights.reshape(1,-1), np.array(y_pred_all)).reshape(-1)

        return np.where(y_pred > 0, 1, -1)
    
    def uncertainty(self, X_test):
        '''
            Provide classification labels for the provided data. For this 
            ensemble scheme, we take a weighted average over predictions.
        '''

        # Get uncertainties from all models.
        y_var_all = []
        for model in self.models:
            y_var = model.uncertainty(X_test)
            y_var_all.append(y_var)

        # Put the uncertainties from each model on the same scale.
        y_var_all = np.transpose(y_var_all)
        y_var_all /= np.max(y_var_all, axis=0)
        y_var_all = np.transpose(y_var_all)

        # Take weighted mean of uncertainties.
        y_var = np.matmul(self.weights.reshape(1,-1), np.array(y_var_all)).reshape(-1)

        return y_var

class ArbitratingEnsemble(AbstractEnsemble):
    ''' 
        Ensemble method that makes predictions, uncertainties based on
        the individual model with the lowest uncertainty prediction.
    '''
    def __init__(self, models):
        super().__init__(models=models)
        self.name        = 'ensemble_arbitrating'

    def train(self, cv=False):
        '''
            Each individual model is trained.
        '''
        for m in self.models:
            m.train(cv=cv)
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided data. For this
            ensemble scheme, we take the predictions of the model with the
            lowest uncertainty value.
        '''

        # Get predictions and uncertainties.
        y_pred_all = []
        for model in self.models:
            y_pred = model.classify(X_test)
            y_pred_all.append(y_pred)
        y_var_all = []
        for model in self.models:
            y_var = model.uncertainty(X_test)
            y_var_all.append(y_var)
        y_pred_all = np.array(y_pred_all)
        y_var_all = np.array(y_var_all)

        # Standardize uncertainty values.
        y_var_all = np.transpose(y_var_all)
        y_var_all /= np.max(y_var_all, axis=0)
        y_var_all = np.transpose(y_var_all)

        # Get predictions of models with the lowest uncertainties.
        y_pred = np.zeros((X_test.shape[0],))
        least_uncertain_indices = np.argmin(y_var_all, axis=0).astype(int)
        for row in range(X_test.shape[0]):
            y_pred[row] = y_pred_all[least_uncertain_indices[row], row]

        return y_pred
    
    def uncertainty(self, X_test):
        '''
            Provide classification labels for the provided data. For this 
            ensemble scheme, we take the lowest uncertainty values.
        '''

        # Get all uncertainties.
        y_var_all = []
        for model in self.models:
            y_var = model.uncertainty(X_test)
            y_var_all.append(y_var)
        y_var_all = np.array(y_var_all)

        # Standardize uncertainty values.
        y_var_all = np.transpose(y_var_all)
        y_var_all /= np.max(y_var_all, axis=0)
        y_var_all = np.transpose(y_var_all)

        return np.min(y_var_all, axis=0)

if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output

    # Get a sample from a dataset.
    dataset = load_data(task='princeton')
    selected_indices = sample(name='medoids', domain=dataset[:,0:-1], size=50, seed=1)
    sample_points = dataset[selected_indices, :]

    # Train a model on the sample.
    model = ArbitratingEnsemble(models=['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard'])
    model.load_data(X=sample_points[:,0:-1], y=sample_points[:,-1])
    model.train(cv=True)
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