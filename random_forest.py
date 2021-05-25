import numpy as np

from sklearn.ensemble import RandomForestClassifier


class RandomForest(object):

    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

        self.model = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                            max_depth=self.max_depth)


    def train(self, X, y):
        self.model.fit(X, y)


    def predict(self, X):
        return self.model.predict(X)


    @staticmethod
    def new(n_estimator=100, criterion='gini', max_depth=None):
        return RandomForest(n_estimators=n_estimator, criterion=criterion, max_depth=max_depth)