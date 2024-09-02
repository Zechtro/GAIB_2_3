import numpy as np


class GaussianNaiveBayes:
    def fit(self, X, y) -> None:
        self.y_train = y
        self.unique_label = np.unique(y)
        self.params = [] 
        for label in self.unique_label:
            temp_X = X[self.y_train== label]
            self.params.append([(features.mean(), features.var()) for features in temp_X.T])

    def likelihood(self, data, mean, var):
        eps = 1e-4  # untuk menghandle case zero division
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((data - mean) ** 2 / (2 * var + eps)))
        return coeff * exponent

    def predict(self, X):
        n_samples, _ = X.shape
        predictions = np.empty(n_samples)
        for idx, feature in enumerate(X):
            posterior_probabilities = []
            for label_idx, label in enumerate(self.unique_label):
                prior_probabilities = np.log((self.y_train== label).mean())

                pairs = zip(feature, self.params[label_idx])
                likelihood = np.sum([np.log(self.likelihood(f, m, v)) for f, (m, v) in pairs])

                posterior_probabilities.append(prior_probabilities + likelihood)

            predictions[idx] = self.unique_label[np.argmax(posterior_probabilities)]

        return predictions

# from sklearn.datasets import load_iris
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.model_selection import train_test_split

# X, y = load_iris(return_X_y=True)
# train_X, test_X, train_y, test_y = train_test_split(
#     X, y, test_size=0.5, random_state=0
# )

# gnb = GaussianNaiveBayes()
# gnb.fit(train_X, train_y)
# predictions = gnb.predict(test_X)

# accuracy = accuracy_score(test_y, predictions)
# precision, recall, fscore, _ = precision_recall_fscore_support(
#     test_y, predictions, average="macro"
# )

# print(f"Accuracy:  {accuracy:.3f}")
# print(f"Precision: {precision:.3f}")
# print(f"Recall:    {recall:.3f}")
# print(f"F-score:   {fscore:.3f}")
# print()
# print(f"Mislabeled points: {(predictions != test_y).sum()}/{test_X.shape[0]}")