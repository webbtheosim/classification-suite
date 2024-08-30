import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from skopt.callbacks import HollowIterationsStopper
from skopt.space import Real, Integer, Categorical

from ClassificationSuite.Models import AbstractModel

class KNN(AbstractModel):
    '''
        Implementation of a k-neighbors classifier.
    '''

    def __init__(self):
        super().__init__()
        self.name    = 'knn'
        self.model   = None
        self.scaler  = None
        self.train_x = None
        self.train_y = None

    def train(self, cv=True):
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

        # If we have enough measurements to do cross-validation,
        # and CV is specified.
        if cv and n_folds > 1:

            # Train SVC with hyperparameter tuning.
            knn_clf = BayesSearchCV(
                estimator=KNeighborsClassifier(),
                search_spaces={
                    'n_neighbors': Integer(1, 20),
                },
                n_iter=20,
                scoring='f1_macro',
                n_jobs=1,
                n_points=1,
                cv=n_folds,
                refit=True,
                random_state=1,
                error_score=0.0
            )
            knn_clf.fit(
                X=train_x, 
                y=self.train_y,
                callback=HollowIterationsStopper(n_iterations=10, threshold=0.03)
            )
            self.model = knn_clf.best_estimator_

        # If we don't have enough measurements to do cross-validation,
        # or CV is not specified.
        else:

            # Train default implementation of SVC.
            if self.model is None:
                n_neighbors = min(5, self.train_x.shape[0])
                self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.model.fit(train_x, self.train_y)
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided
            data.
        '''

        # Scale data.
        X_test = self.scaler.transform(X_test)

        return self.model.predict(X_test)
    
    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided
            data.
        '''

        # Scale data.
        X_test = self.scaler.transform(X_test)
        logits = self.model.predict_proba(X_test)

        return entropy(logits, axis=1)
    
if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output

    # Get a sample from a dataset.
    dataset = load_data(task='glotzer_pf')
    selected_indices = sample(name='medoids', domain=dataset[:,0:-1], size=3, seed=1)
    sample_points = dataset[selected_indices,:]

    # Train a model on the sample.
    model = KNN()
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