import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.semi_supervised import LabelPropagation
from skopt import BayesSearchCV
from skopt.callbacks import HollowIterationsStopper
from skopt.space import Real, Integer, Categorical
import warnings

from ClassificationSuite.Models import AbstractModel

# Silence Convergence and Runtime warnings associated with
# fitting the LP model.
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class LP(AbstractModel):
    '''
        Implementation of the label propagation
        semi-supervised learning algorithm. A train
        method is not needed based on the implementation
        in sklearn.

        Note: Given the peculiarity of the sklearn implementation,
        make sure that the model has always been used to 'classify'
        before it is used to get 'uncertainty.'
    '''

    def __init__(self):
        super().__init__()
        self.name    = 'lp'
        self.scaler  = None
        self.train_x = None
        self.train_y = None
        self.model   = None

    def classify(self, X_test, cv=True):
        '''
            Provide classification labels for the provided
            data.
        '''

        # Construct a full dataset with labels 0/1 and -1 is
        # reserved for unlabelled points.
        train_y = np.where(self.train_y == -1, 0, 1)
        y = -1.0 * np.ones(X_test.shape[0])
        for index1 in range(self.train_x.shape[0]):
            for index2 in range(X_test.shape[0]):
                if (self.train_x[index1,:] == X_test[index2]).all():
                    y[index2] = train_y[index1]
        
        # Scale domain.
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_test)

        # Use hyperparameter tuning to find number of neighbors.
        if cv:

            # Define HT space.
            n_neighbors = [1, 3, 5, 7, 10, 20]
            
            # Get indices of test points.
            X_unlabelled = X_scaled[np.where(y == -1)[0]]
            X_labelled = X_scaled[np.where(y != -1)[0]]
            y_unlabelled = y[np.where(y == -1)[0]]
            y_labelled = y[np.where(y != -1)[0]]

            # Determine number of CV folds.
            class_1_indices = np.where(y == 1)[0]
            class_0_indices = np.where(y == 0)[0]
            n_folds = np.min([len(class_1_indices), len(class_0_indices), 5])
            
            # Assuming we have measured an adequate number of points
            # for cross-validation.
            if n_folds > 1:

                # Get CV fold indices.
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

                # Train LP model on data.
                cv_results = {}
                for index, (train_index, val_index) in enumerate(skf.split(X_labelled, y_labelled)):

                    # Combine appropriate training points with larger domain.
                    X_train = X_labelled[train_index]
                    y_train = y_labelled[train_index]
                    X_val = X_labelled[val_index]
                    y_val = y_labelled[val_index]

                    # Add unlabelled points to training data.
                    X_train = np.vstack((X_train, X_unlabelled))
                    y_train = np.hstack((y_train, y_unlabelled))

                    # Train models for different numbers of neighbors.
                    for n in n_neighbors:
                        temp_model = LabelPropagation(kernel='knn', n_neighbors=n)
                        temp_model.fit(X_train, y_train)
                        y_pred = temp_model.predict(X_val)

                        # Evaluate using Macro F1 score.
                        temp_score = f1_score(y_val, y_pred, average='macro')

                        # Save CV results.
                        if n not in cv_results.keys():
                            cv_results[n] = []
                        cv_results[n].append(temp_score)

                # Determine optimal n_neighbors value.
                cv_results_sorted = dict(sorted(cv_results.items(), key=lambda item: -np.mean(item[1])))
                n_optimal = list(cv_results_sorted.keys())[0]

                # Retrain on the entire training dataset.
                self.model = LabelPropagation(kernel='knn', n_neighbors=n_optimal)
                self.model.fit(X_scaled, y)

            # Just use default implementation if an inadequate number
            # of points have been measured.
            else:
                self.model = LabelPropagation(kernel='knn', n_neighbors=7)
                self.model.fit(X_scaled, y)

        # Just fit a model using the existing parameter set.
        else:
            if self.model is None:
                self.model = LabelPropagation(kernel='knn', n_neighbors=12)
            self.model.fit(X_scaled, y)

        # Get classification predictions.
        y_pred = self.model.predict(X=X_scaled)

        return np.where(y_pred == 0, -1, 1)
    
    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided
            data. Uncertainties are based on the 'LC'
            metric shown to perform well by Terayama et al.
        '''
        X_scaled = self.scaler.transform(X_test)
        logits = self.model.predict_proba(X_scaled)
        return 1.0 - np.max(logits, axis=1)
    
    def recommend(self, domain, size, chosen_indices):
        '''
            Recommend 'size' number of points to be
            measured next from the available 'domain.'
            Points in the training set should not be
            considered for further measurement. This method
            differs from the AbstractModel class because of
            the sklearn LP implementation.
        '''
        
        sample = []
        for _ in range(size):

            # Compute acquisition function per Dai and Glotzer.
            y_pred = self.classify(X_test=domain, cv=False)
            y_acq = self.uncertainty(X_test=domain)
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

        return sample
    
    def score(self, X_test, y_test):
        '''
            Evaluate the current model on new data.
            Should return balanced accuracy, Macro
            F1 score, and Matt. corr. coefficient.
        '''
        y_pred = self.classify(X_test=X_test, cv=False)
        scores = [
            balanced_accuracy_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred, average='macro'),
            matthews_corrcoef(y_true=y_test, y_pred=y_pred)
        ]

        return scores

if __name__ == '__main__':

    from ClassificationSuite.Tasks.utils import load_data
    from ClassificationSuite.Samplers.samplers import sample
    from ClassificationSuite.Models.utils import visualize_model_output

    # Get a sample from a dataset.
    dataset = load_data(task='glotzer_xa')
    selected_indices = sample(name='random', domain=dataset[:,0:-1], size=3, seed=6)
    sample_points = dataset[selected_indices, :]

    # Train a model on the sample.
    model = LP()
    model.load_data(X=sample_points[:,0:-1], y=sample_points[:,-1])
    model.train()
    y_pred = model.classify(X_test=dataset[:,0:-1], cv=True)
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