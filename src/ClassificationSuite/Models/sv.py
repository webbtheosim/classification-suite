import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.callbacks import HollowIterationsStopper
from skopt.space import Real, Integer, Categorical

from ClassificationSuite.Models import AbstractModel

class SV(AbstractModel):
    '''
        Implementation of a support vector classifier.
    '''

    def __init__(self):
        super().__init__()
        self.name    = 'sv'
        self.model   = None
        self.scaler  = None
        self.train_x = None
        self.train_y = None
        self.explore = False

    def train(self, cv=True, cv_score=False):
        '''
            Fit model to the loaded training data. This
            should involve hyperparameter tuning. This
            method should also deal with label changes
            or situations where only one class has been
            identified.
        '''

        # Scale training data.
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_x)
        train_x = self.scaler.transform(self.train_x)

        # Decide on cross-validation, exploration, etc.
        class_1 = np.where(self.train_y == 1)[0]
        class_0 = np.where(self.train_y == -1)[0]
        n_folds = np.min([len(class_1), len(class_0), 5])

        # If only one class has been discovered, this
        # causes problems with SVC. No model is trained.
        if len(class_0) == 0 or len(class_1) == 0:
            self.explore = True

            # If only one class has been discovered, assume that 
            # there would be perfect CV prediction.
            if cv_score:
                return 1.00

        # If we have enough measurements to do cross-validation,
        # and CV is specified.
        elif cv and n_folds > 1:

            # Reset explore value.
            self.explore = False

            # Train SVC with hyperparameter tuning.
            svc_clf = BayesSearchCV(
                estimator=SVC(probability=True),
                search_spaces={
                    'C': Real(1e-2, 1e2, prior='log-uniform'),
                    'kernel': Categorical(['rbf', 'linear', 'sigmoid', 'poly']),
                    'degree': Integer(1,5),
                    'gamma': Categorical(['scale', 'auto']),
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
            svc_clf.fit(
                X=train_x, 
                y=self.train_y,
                callback=HollowIterationsStopper(n_iterations=10, threshold=0.03)
            )
            self.model = svc_clf.best_estimator_

            # If a cv_score is requested.
            if cv_score:
                return np.max(svc_clf.cv_results_['mean_test_score'])

        # If we don't have enough measurements to do cross-validation,
        # or CV is not specified.
        else:

            # Reset explore value.
            self.explore = False

            # Train default implementation of SVC.
            if self.model is None:
                self.model = SVC(probability=True)
            self.model.fit(train_x, self.train_y)

            # If a cv_score is requested, but we only have have one 
            # element in the minority class.
            if cv_score:
                return 0.8
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided
            data.
        '''

        # Scale data.
        X_test = self.scaler.transform(X_test)

        # If only one class has been discovered.
        if self.explore:
            label = self.train_y[0]
            return label * np.ones(shape=(X_test.shape[0],))

        return self.model.predict(X_test)
    
    def predict(self, X_test):
        '''
            Provide classification probabilities for the provided data
            on a scale from -1 to 1.
        '''
        if self.explore:
            return np.ones(shape=(X_test.shape[0],))
        X_test = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test)
        logits = np.max(self.model.predict_proba(X_test), axis=1)
        return logits * y_pred + (1.0 - logits) * y_pred * -1.0
    
    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided
            data.
        '''

        # Scale data.
        X_test = self.scaler.transform(X_test)
        train_x = self.scaler.transform(self.train_x)

        # If only one class has been discovered, assign
        # high uncertainties to those values which are
        # furthest from the current training data.
        if self.explore:
            distances = cdist(X_test, train_x)
            return np.min(distances, axis=1)

        logits = self.model.predict_proba(X_test)
        return entropy(logits, axis=1)
    
if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output

    # Get a sample from a dataset.
    dataset = load_data(task='glotzer_pf')
    selected_indices = sample(name='medoids', domain=dataset[:,0:-1], size=100, seed=6)
    sample_points = dataset[selected_indices,:]

    # Train a model on the sample.
    model = SV()
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