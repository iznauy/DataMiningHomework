import numpy as np

from sklearn.ensemble import RandomForestClassifier


class RandomForest(object):

    def __init__(self, n_estimator=100, criterion='gini', max_depth=None):
        self.n_estimator = n_estimator
        self.criterion = criterion
        self.max_depth = max_depth

        self.model = RandomForestClassifier(n_estimators=self.n_estimator, criterion=self.criterion,
                                            max_depth=self.max_depth)


    def train(self, X, y):
        self.model.fit(X, y)


    def predict(self, X):
        return self.model.predict(X)


    def clear(self):
        self.model = RandomForestClassifier(n_estimators=self.n_estimator, criterion=self.criterion,
                                            max_depth=self.max_depth)
