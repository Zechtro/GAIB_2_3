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
        approx = np.dot(X, self.w) - self.b
        approx[approx < 0] = 0
        return np.sign(approx)

# import pandas as pd
# # df = pd.read_csv("dataset/tes.csv")
# df = pd.read_csv("dataset/SleepyDriverEEGBrainwave.csv")
# y = df["classification"]
# X = df.drop("classification", axis=1)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# model = SVM()
# model.fit(X_train, y_train)

# y_predict = model.predict(X_test)
# print(y_predict)

# def accuracy(y_true, y_pred):
#         accuracy = np.sum(y_true == y_pred) / len(y_true)
#         return accuracy

# print("SVM classification accuracy", accuracy(y_test, y_predict))

# import matplotlib.pyplot as plt
# def visualize_svm(X, y):
#     def get_hyperplane_value(x, w, b, offset):
#         return (-w[0] * x + b + offset) / w[1]

#     X = X.values
#     y = y.values
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

#     x0_1 = np.amin(X[:, 0])
#     x0_2 = np.amax(X[:, 0])

#     x1_1 = get_hyperplane_value(x0_1, model.w, model.b, 0)
#     x1_2 = get_hyperplane_value(x0_2, model.w, model.b, 0)

#     x1_1_m = get_hyperplane_value(x0_1, model.w, model.b, -1)
#     x1_2_m = get_hyperplane_value(x0_2, model.w, model.b, -1)

#     x1_1_p = get_hyperplane_value(x0_1, model.w, model.b, 1)
#     x1_2_p = get_hyperplane_value(x0_2, model.w, model.b, 1)

#     ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
#     ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
#     ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

#     x1_min = np.amin(X[:, 1])
#     x1_max = np.amax(X[:, 1])
#     ax.set_ylim([x1_min - 3, x1_max + 3])

#     plt.show()

# # visualize_svm(X, y)
# from sklearn.svm import SVC
# svm_model = SVC(kernel="linear")  # You can customize parameters like kernel, C, gamma, etc.
# svm_model.fit(X_train, y_train)
# y_pred = svm_model.predict(X_test)
# print("SVM classification accuracy", accuracy(y_test, y_pred))