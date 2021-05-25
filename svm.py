import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class SVM(object):

    def __init__(self, C = 1.0, kernel='linear'):
        self.C = C
        self.kernel = kernel

        self.model = make_pipeline(StandardScaler(), SVC(C=self.C, kernel=self.kernel))


    def train(self, X, y):
        self.model.fit(X, y)


    def predict(self, X):
        return self.model.predict(X)


    @staticmethod
    def new(C = 1.0, kernel='linear'):
        return SVM(C=C, kernel=kernel)