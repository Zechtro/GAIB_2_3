import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self, k=3, max_iteration=100):
        self.k = k
        self.max_iteration = max_iteration
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, X):
        self.X = X.values
        self.features = X.columns
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iteration):
            self.clusters = self._create_clusters(self.centroids)
            old_centroids = self.centroids
            self.centroids = self._update_centroids(self.clusters)

            if self._is_converged(old_centroids, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels


    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _update_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, centroids):
        distances = [euclidean_distance(old_centroids[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def plot2D(self,feature1, feature2):
        i = self.features.get_loc(feature1)
        j = self.features.get_loc(feature2)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        for _, cluster_idx in enumerate(self.clusters):
            points = self.X[cluster_idx]
            ax.scatter(points[:, i], points[:, j])

        for point in self.centroids:
            ax.scatter(point[i], point[j],marker="x", color="red", linewidth=3)

        plt.show()
        
    def plot3D(self,feature1, feature2, feature3):
        i = (self.features).get_loc(feature1)
        j = (self.features).get_loc(feature2)
        k = (self.features).get_loc(feature3)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_zlabel(feature3)

        for _, cluster_idx in enumerate(self.clusters):
            points = self.X[cluster_idx]
            ax.scatter(points[:, i], points[:, j], points[:, k])

        for point in self.centroids:
            ax.scatter(point[i], point[j], point[k], marker="x", color="red", linewidth=10)

        plt.show()


# import pandas as pd
# df = pd.read_csv("dataset/SleepyDriverEEGBrainwave.csv")
# y = df["classification"].values
# X = df.drop("classification", axis=1)

# print(X.head())

# model = KMeans(k=2)
# predictions = model.predict(X)
# features = X.columns
# features = features.drop("delta")
# len_feat = len(features)
# print(features)
# # feature visualisasi bagus: delta
# for i in range(len_feat):
#     for j in range(len_feat):
#         if(i!=j):
#             # model.plot2D(features[i], features[j])
#             model.plot3D("delta", features[i], features[j])
