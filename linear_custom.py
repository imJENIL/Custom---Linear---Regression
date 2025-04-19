import numpy as np

class Linear:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):

            y_pred = np.dot(x, self.weights) + self.bias

            dw = (1/n_samples)*np.dot(x.T, (y_pred - y))
            db = (1/n_samples)*np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db

        return self

    def pred(self, x):
        return np.dot(x, self.weights) + self.bias