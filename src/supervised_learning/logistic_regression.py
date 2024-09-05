import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, learning_rate=0.001, n_iteration=1000, regularization="", lambd=0.01):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.regularization = regularization
        self.lambd = lambd
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = X.values
        y = y.values
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iteration):
            linear_prediction = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_prediction)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)
            
            if(self.regularization == 'l1'):
                dw += (self.lambd / n_samples) * np.sign(self.weights)
            elif(self.regularization == 'l2'):
                dw += (self.lambd / n_samples) * self.weights
            elif(self.regularization == ""):
                dw = dw
            else:
                raise ValueError('Invalid regularization, should either be: "l1", "l2", or "".')

            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db

    def predict(self, X):
        X = X.values
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_prediction)
        # kalo y = 0.5 lebih baik anggep jadi 1 (mengantuk) untuk mengantisipasi
        final_pred = [0 if y<0.5 else 1 for y in y_pred]
        return final_pred