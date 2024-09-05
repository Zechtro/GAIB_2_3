import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, C=1, n_iteration=1000):
        self.lr = learning_rate
        self.C = C
        self.n_iteration = n_iteration
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = X.values
        y = y.values
        y[y == 0] = -1
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iteration):
            for idx, x_i in enumerate(X):
                margin = y_[idx] * (np.dot(x_i, self.w) - self.b)
                if(margin>=1):
                    self.w -= self.lr * (self.C * self.w)
                else:
                    self.w -= self.lr * (self.C * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        X = X.values
        approx = np.dot(X, self.w) - self.b
        approx[approx < 0] = 0
        return np.sign(approx)