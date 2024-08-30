import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.callbacks import HollowIterationsStopper
from skopt.space import Real, Integer, Categorical

from ClassificationSuite.Models import AbstractModel

class RF(AbstractModel):
    '''
        Implementation of a random forest classifier.
        The uncertainties of the random forest are based
        on the Shannon entropy of the probability of
        classification, per Telleria-Allika et al. (2022).
    '''
    
    def __init__(self):
        super().__init__()
        self.name    = 'rf'
        self.model   = None
        self.train_x = None
        self.train_y = None

    def train(self, cv=True, cv_score=False):
        '''
            Fit model to the loaded training data. This
            should involve hyperparameter tuning. This
            method should also deal with label changes
            or situations where only one class has been
            identified.
        '''

        # Determine number of CV folds.
        class_1 = np.where(self.train_y == 1)[0]
        class_2 = np.where(self.train_y == -1)[0]
        n_folds = np.min([len(class_1), len(class_2), 5])
        
        # If cross-validation is specified.
        if cv and n_folds > 1:

            # Define parameters for cross validation.
            rfc_clf = BayesSearchCV(
                estimator=RandomForestClassifier(),
                search_spaces={
                    'n_estimators': Integer(100, 300),
                    'max_features': Categorical(['sqrt', 'log2', None]),
                    'max_depth': Integer(10, 100),
                    'min_samples_split': Integer(2,6),
                    'min_samples_leaf': Integer(1,4) 
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
            rfc_clf.fit(
                X=self.train_x, 
                y=self.train_y, 
                callback=HollowIterationsStopper(n_iterations=10, threshold=0.03)
            )
            self.model = rfc_clf.best_estimator_

            # Return CV score if specified.
            if cv_score:
                return np.max(rfc_clf.cv_results_['mean_test_score'])

        # If cross-validation is not specified or there is not enough
        # measured data for training.
        else:
            if self.model is None:
                self.model = RandomForestClassifier()
            self.model.fit(X=self.train_x, y=self.train_y)

            # Report a CV score if requested.
            if cv_score:

                # If only one class, assume perfect prediction.
                if len(class_1) == 0 or len(class_2) == 0:
                    return 1.00
                
                # If one member of one class is found, assume it would
                # be predicted incorrectly for that fold in 5-fold CV.
                else:
                    return 0.8
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided
            data.
        '''
        return self.model.predict(X=X_test)
    
    def predict(self, X_test):
        '''
            Provide classification probabilities for the provided data
            on a scale from -1 to 1.
        '''
        y_pred = self.model.predict(X=X_test)
        logits = np.max(self.model.predict_proba(X=X_test), axis=1)
        return logits * y_pred + (1.0 - logits) * y_pred * -1.0
    
    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided
            data.
        '''
        logits = self.model.predict_proba(X=X_test)
        return entropy(logits, axis=1)
    
if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output

    # Get a sample from a dataset.
    dataset = load_data(task='glotzer_pf')
    selected_indices = sample(name='medoids', domain=dataset[:,0:-1], size=100, seed=1)
    sample_points = dataset[selected_indices, :]

    # Train a model on the sample.
    model = RF()
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