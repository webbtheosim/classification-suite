import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

class AbstractModel:
    ''' 
        Abstract class for all models. Shows
        the API followed by all implemented 
        models.
    '''

    def __init__(self):
        self.name = 'model'

    def load_data(self, X, y):
        '''
            Save training data at the class level.
            If called repeatedly, the training data
            will be overridden.
        '''
        self.train_x = X
        self.train_y = y.reshape(-1)

    def add_data(self, X_new, y_new):
        '''
            Add new training data to an existing set.
        '''
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

    def train(self, cv=False, cv_score=False):
        '''
            Fit model to the loaded training data. This
            should involve hyperparameter tuning. This
            method should also deal with label changes
            or situations where only one class has been
            identified.
        '''
        return 0
    
    def classify(self, X_test):
        '''
            Provide classification labels for the provided data.
        '''
        return 0
    
    def predict(self, X_test):
        '''
            Provide classification probabilities for the provided data
            on a scale from -1 to 1.
        '''
        return 0
    
    def uncertainty(self, X_test):
        '''
            Provide uncertainty values for the provided data.
        '''
        return 0
    
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
            y_pred = self.classify(X_test=domain)
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

            # Retrain model.
            self.train(cv=False)

        return sample

    def score(self, X_test, y_test):
        '''
            Evaluate the current model on new data.
            Should return balanced accuracy, Macro
            F1 score, and Matt. corr. coefficient.
        '''
        y_pred = self.classify(X_test=X_test)
        scores = [
            balanced_accuracy_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred, average='macro'),
            matthews_corrcoef(y_true=y_test, y_pred=y_pred)
        ]

        return scores