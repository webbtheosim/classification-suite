import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from skopt import BayesSearchCV
from skopt.callbacks import HollowIterationsStopper
from skopt.space import Real, Integer, Categorical
from xgboost import XGBClassifier

from ClassificationSuite.Models import AbstractModel

class XGB(AbstractModel):
    '''
        Implementation of extreme gradient boosted decision
        trees (XGBoost). Uncertainties are determined using
        a bagging ensemble of XGBoost models.
    '''
    
    def __init__(self, n_models=10):
        super().__init__()
        self.name     = 'xgb'
        self.model    = None
        self.n_models = n_models
        self.models   = []
        self.train_x  = None
        self.train_y  = None
        self.explore  = False

    def train(self, cv=True, cv_score=False):
        '''
            Fit model to the loaded training data. This
            should involve hyperparameter tuning. This
            method should also deal with label changes
            or situations where only one class has been
            identified.
        '''

        # Labels are in terms of 0/1, instead of arbitrary.
        train_y = np.where(self.train_y == -1, 0, 1)

        # Determine number of CV folds.
        class_1 = np.where(train_y == 1)[0]
        class_2 = np.where(train_y == 0)[0]
        n_folds = np.min([len(class_1), len(class_2), 5])

        # If only one class has been discovered, this
        # causes problems with XGB. No model is trained.
        if len(class_1) == 0 or len(class_2) == 0:
            self.explore = True

            # If only one class, assume perfect prediction.
            if cv_score:
                return 1.00
    
        # If cross-validation is specified.
        elif cv and n_folds > 1:

            # Reset explore value.
            self.explore = False

            # Define parameters for cross validation.
            xgb_clf = BayesSearchCV(
                estimator=XGBClassifier(objective='binary:logistic'),
                search_spaces={
                    'eta': Real(1e-2, 1.0, prior='log-uniform'),
                    'max_depth': Integer(3, 15),
                    'lambda': Real(1e-2, 1e2, prior='log-uniform'),
                },
                n_iter=100,
                scoring='f1_macro',
                n_jobs=1,
                n_points=1,
                cv=n_folds,
                refit=True,
                verbose=0,
                random_state=1,
                error_score=0.0
            )
            xgb_clf.fit(
                X=self.train_x, 
                y=train_y, 
                callback=HollowIterationsStopper(n_iterations=10, threshold=0.03)
            )
            self.model = xgb_clf.best_estimator_
            self.config = xgb_clf.best_params_

            # Train individual members of the ensemble using bootstrapping.
            self.models = []
            indices = [i for i in range(self.train_x.shape[0])]
            for _ in range(self.n_models):
                train_indices = np.random.choice(indices, size=int(0.7 * self.train_x.shape[0]))

                # Double-check to make sure train_indices includes indices from both
                # classes. If not, manually add one example from the minority class.
                train_1 = np.where(train_y[train_indices] == 1)[0]
                train_2 = np.where(train_y[train_indices] == 0)[0]
                if len(train_1) == 0:
                    train_indices[-1] = class_1[0]
                if len(train_2) == 0:
                    train_indices[-1] = class_2[0]

                model = XGBClassifier(**self.config)
                model.fit(X=self.train_x[train_indices], y=train_y[train_indices])
                self.models.append(model)

            # Return CV score if specified.
            if cv_score:
                return np.max(xgb_clf.cv_results_['mean_test_score'])

        # If cross-validation is not possible with the
        # number of points that have been measured.
        else:

            # Reset explore value.
            self.explore = False

            # Instantiate models if they haven't been already.
            if len(self.models) == 0:
                for _ in range(self.n_models):
                    self.models.append(XGBClassifier())
            if self.model is None:
                self.model = XGBClassifier()

            # Train ensemble of models for uncertainty prediction.
            indices = [i for i in range(self.train_x.shape[0])]
            for m in self.models:
                train_indices = np.random.choice(indices, size=int(0.7 * self.train_x.shape[0]))

                # Double-check to make sure train_indices includes indices from both
                # classes. If not, manually add one example from the minority class.
                train_1 = np.where(train_y[train_indices] == 1)[0]
                train_2 = np.where(train_y[train_indices] == 0)[0]
                if len(train_1) == 0:
                    train_indices[-1] = class_1[0]
                if len(train_2) == 0:
                    train_indices[-1] = class_2[0]

                m.fit(X=self.train_x[train_indices], y=train_y[train_indices])

            # Train individual model for classification.
            self.model.fit(X=self.train_x, y=train_y)

            # If one member of one class is found, assume it would
            # be predicted incorrectly for that fold in 5-fold CV.
            if cv_score:
                return 0.8
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided
            data.
        '''
        if self.explore:
            label = self.train_y[0]
            return label * np.ones(shape=(X_test.shape[0],))
        return np.where(self.model.predict(X_test) > 0.5, 1, -1)
    
    def predict(self, X_test):
        '''
            Provide classification probabilities for the provided data
            on a scale from -1 to 1.
        '''
        if self.explore:
            return np.ones(shape=(X_test.shape[0],))
        y_pred_all = []
        for model in self.models:
            y_pred = model.predict(X_test)
            y_pred_all.append(y_pred)
        return np.mean(y_pred_all, axis=0)
    
    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided
            data.
        '''

        # If only one class has been discovered, assign
        # high uncertainties to those values which are
        # furthest from the current training data.
        if self.explore:
            distances = cdist(X_test, self.train_x)
            return np.min(distances, axis=1)
        
        y_pred_all = []
        for model in self.models:
            y_pred = model.predict(X_test)
            y_pred_all.append(y_pred)
        return np.std(y_pred_all, axis=0)
    
if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output

    # Get a sample from a dataset.
    dataset = load_data(task='glotzer_pf')
    selected_indices = sample(name='medoids', domain=dataset[:,0:-1], size=100, seed=1)
    sample_points = dataset[selected_indices, :]

    # Train a model on the sample.
    model = XGB()
    model.load_data(X=sample_points[:,0:-1], y=sample_points[:,-1])
    model.train()
    y_pred = model.classify(X_test=dataset[:,0:-1])
    y_acq = model.uncertainty(X_test=dataset[:,0:-1])
    y_acq = model.predict(X_test=dataset[:,0:-1])

    # Get scores.
    print(f'Scores: {model.score(X_test=dataset[:,0:-1], y_test=dataset[:,-1])}')

    # Visualize predictions of model.
    visualize_model_output(
        dataset=dataset,
        chosen_points=sample_points[:,0:-1], 
        y_pred=y_pred, 
        y_acq=y_acq
    )