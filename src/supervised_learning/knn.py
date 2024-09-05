import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum(((x1)-(x2))**2))
    return distance

def manhattan_distance(x1, x2):
    return np.sum(np.abs((x1)-(x2)))

def minkowski_distance(x1, x2, p=2):
    return np.power(np.sum(np.abs((x1)-(x2))**p), 1/p)

def avgDistance(label, k_nearest_labels, k_indices, distances): # Digunakan untuk menghandle ketika most common label memiliki jumlah yang sama
    return np.mean([distances[k_indices[i]] for i in (np.where(k_nearest_labels == label)[0])])

class KNN:
    def __init__(self, k=3, distance_metric="euclidean", minkowski_p=2):
        self.k = k
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p

    def fit(self, X, y):
        self.X_train = X.values
        self.y_train = y.values

    def predict(self, X):
        X = X.values
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # print(self.X_train[:5])
        if(self.distance_metric == "euclidean"):
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif(self.distance_metric == "manhattan"):
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        elif(self.distance_metric == "minkowski"):
            distances = [minkowski_distance(x, x_train, self.minkowski_p) for x_train in self.X_train]
        else:
            raise ValueError("Invalid metric distances, should either be: euclidean, manhattan, or minkowski.")
            
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common()
        max_count = most_common[0][1]

        most_common_labels = [value for value, count in most_common if count == max_count]
        
        if(len(most_common_labels) > 1):
            labels_avgDistance = [avgDistance(label, k_nearest_labels, k_indices, distances) for label in most_common_labels]
            nearest_label_idx = (np.argsort(labels_avgDistance)[:1])[0]
            return most_common_labels[nearest_label_idx]
        else:
            return most_common_labels[0]