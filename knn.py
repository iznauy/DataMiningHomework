import numpy as np

from sklearn.preprocessing import StandardScaler

class KNN(object):

    def __init__(self, k=10):
        self.k = k
        self.X = None
        self.y = None
        self.scaler = None


    def train(self, X, y):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X)
        self.y = y


    def predict(self, X):
        X = self.scaler.transform(X)
        y = []
        for x in X:
            y.append(np.argmax(np.bincount(self.y[np.argsort(np.sqrt(np.sum((self.X - x) ** 2, axis=1)))[:self.k]])))
        return np.array(y)


    def clear(self):
        self.X = None
        self.y = None
        self.scaler = None
