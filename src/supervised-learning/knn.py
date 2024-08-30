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
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
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

# # Test 1   
# X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y_train = np.array([0, 0, 1, 1])
# knn = KNN(k=3, distance_metric="manhattan")
# knn.fit(X_train, y_train)

# # Test with different input points
# X_test = np.array([[1.5, 2.5], [3.5, 4.5], [2, 4]])
# expected_predictions = np.array([0, 1, 1])
# predictions = knn.predict(X_test)

# print("EXPECT:", expected_predictions)
# print("PRED:", expected_predictions)

# # Test 2
# X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
# y_train = np.array([0, 0, 1, 1, 1])
# knn = KNN(k=4, distance_metric="minkowski", minkowski_p=5)
# knn.fit(X_train, y_train)
# X_test = np.array([[3, 3]])
# expected_prediction = 1  # May vary depending on tie-breaking method
# prediction = knn.predict(X_test)[0]
# print("Test 2 - Tie-Breaking (Equal Distances):")
# print("Expected:", expected_prediction)
# print("Predicted:", prediction)