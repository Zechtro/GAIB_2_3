import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum(((x1)-(x2))**2))
    return distance

def manhattan_distance(x1, x2):
    return np.sum(np.abs((x1)-(x2)))

def minkowski_distance(x1, x2, p=2):
    return np.power(np.sum(np.abs((x1)-(x2))**p), 1/p)

class Point:
    def __init__(self, coordinates, type, neighboringPoints, cluster):
        self.coordinates = coordinates
        self.type = type #1 for core point, 2 for border point, 3 for outlier
        self.neighboringPoints = neighboringPoints
        self.cluster = cluster #1, 2, 3, ... when it has been assigned to a cluster

class DBSCAN:
    def __init__(self, epsilon=0.2, min_samples=3, distance_metric="euclidean"):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.distance_metric = distance_metric
    
    def _get_neighbor(self, p):
        NeighboringPoints = []
        type = 0
        for j in range(len(self.X)):
            point = self.X[j]
            if(self.distance_metric == "euclidean"):
                dist = euclidean_distance(point, p)
            elif(self.distance_metric == "manhattan"):
                dist = manhattan_distance(point, p)
            elif(self.distance_metric == "minkowski"):
                dist = minkowski_distance(point, p)
            else:
                raise ValueError("Invalid metric distances, should either be: euclidean, manhattan, or minkowski.")
            
            if (dist < self.epsilon):
                NeighboringPoints.append(j)
                
        if (len(NeighboringPoints) > self.min_samples):
            type = 1 #core point
        elif (len(NeighboringPoints) > 1):
            type = 2 #border point
        else:
            type = 3 #outlier
        return [NeighboringPoints, type]

    def predict(self, X):
        self.X = X.values
        self.features = X.columns
        currentCluster = 0 
        points = []
        for data in self.X: #Step 1: Initialize points as core, border, or outlier points; O(n^2)
            [NeighboringPoints, type] = self._get_neighbor(data)
            points.append(Point(data, type, NeighboringPoints, 1 - type))
        for i in range(len(self.X)):
            if points[i].cluster == 0: #if a point is a core point and it has not been clustered yet
                                    #Since there are log(n) clusters, you encounter an unclustered core point log(n) times
                currentCluster = currentCluster + 1
                points[i].cluster = currentCluster
                self.findClusterPoints(currentCluster, points, i)
        self.df_result = pd.concat([X, pd.DataFrame([point.cluster for point in points], columns=["cluster"])],axis=1)
        return self.df_result
    
    def findClusterPoints(self, currentCluster, points, position): #Step 2b; [log(n/log(n))]^[log(n/log(n))]
        ClusterMembers = points[position].neighboringPoints
        i = 0
        while (i < len(ClusterMembers)):
            expansionPoint = ClusterMembers[i] #set an expansion point
            if (points[expansionPoint].cluster == -1): #if it's a border point that has NOT been assigned to a cluster
                points[expansionPoint].cluster = currentCluster
            elif (points[expansionPoint].cluster == 0): #if it's a core point that has NOT been assigned to a cluster
                points[expansionPoint].cluster = currentCluster
                ClusterMembers = ClusterMembers + points[expansionPoint].neighboringPoints
            i = i + 1
    
    def plot2D(self,feature1, feature2):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        for cluster in self.df_result["cluster"].unique():
            points = self.df_result.loc[self.df_result['cluster'] == cluster]
            ax.scatter(points[feature1], points[feature2])

        plt.show()
        
    def plot3D(self,feature1, feature2, feature3):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_zlabel(feature3)

        for cluster in self.df_result["cluster"].unique():
            points = self.df_result.loc[self.df_result['cluster'] == cluster]
            ax.scatter(points[feature1], points[feature2], points[feature3])

        plt.show()
        
        
# # ============================================
# # from sklearn.datasets import make_blobs
# # import seaborn as sns
# # centers = [[1, 1], [-1, -1], [1, -1]]
# # X, labels_true = make_blobs(
# #     n_samples=1000, centers=centers, cluster_std=0.4, random_state=0
# # )
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# df = pd.read_csv("dataset/SleepyDriverEEGBrainwave.csv")
# y = df["classification"].values
# X = df.drop("classification", axis=1)
# X_scaled = StandardScaler().fit_transform(X.values).copy()
# X = pd.DataFrame(X_scaled, columns=X.columns)
# model = DBSCAN(epsilon=1.2)
# df_final = model.predict(pd.DataFrame(X))
# print(df_final.head())
# print(df_final["cluster"].unique())
# for feature in df_final.columns:
#     for feature2 in df_final.columns:
#         if(feature!=feature2 and feature != "cluster" and feature2 != "cluster"):
#             model.plot2D(feature, feature2)
#             model.plot3D("delta", feature, feature2)