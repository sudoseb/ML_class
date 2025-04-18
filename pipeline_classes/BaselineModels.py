from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import numpy as np

class BaselineModel:
    def __init__(self, task_type='regression', strategy='mean', random_state=None):
        """
        task_type: 'regression' or 'classification'
        strategy: 
            - For regression: 'mean', 'median'
            - For classification: 'most_frequent', 'stratified', 'uniform', 'constant'
        """
        self.task_type = task_type
        self.strategy = strategy
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def fit(self, X, y):
        if self.task_type == 'regression':
            self.model = DummyRegressor(strategy=self.strategy)
        elif self.task_type == 'classification':
            self.model = DummyClassifier(strategy=self.strategy, random_state=self.random_state)
        else:
            raise ValueError("task_type must be either 'regression' or 'classification'")
        
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction.")
        return self.model.predict(X)

    def score(self, X, y, metrics=None):
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before scoring.")

        y_pred = self.predict(X)

        if self.task_type == 'regression':
            if metrics is None:
                metrics = ['rmse', 'mae']
            scores = {}
            if 'rmse' in metrics:
                scores['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            if 'mae' in metrics:
                scores['mae'] = np.mean(np.abs(y - y_pred))
            return scores

        elif self.task_type == 'classification':
            if metrics is None:
                metrics = ['accuracy', 'f1']
            scores = {}
            if 'accuracy' in metrics:
                scores['accuracy'] = accuracy_score(y, y_pred)
            if 'f1' in metrics:
                scores['f1'] = f1_score(y, y_pred, average='weighted')
            return scores
