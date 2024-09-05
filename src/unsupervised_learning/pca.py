import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        X = X.values
        self.mean = np.mean(X, axis=0)
        X = X -  self.mean
        cov = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.all_components = eigenvectors
        self.components = eigenvectors[:self.n_components]
        
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_variance

    def transform(self, X):
        X = X.values
        X = X - self.mean
        return np.dot(X, self.components.T)
    
    def show_explained_variance_ratio(self):
        print("="*8,"Explained Variance Ratio","="*8)
        for i in range(len(self.components)):
            print(f"PC{i+1}:", self.explained_variance_ratio[i])
    
    def show_all_explained_variance_ratio(self):
        print("="*8,"Explained Variance Ratio","="*8)
        for i in range(len(self.all_components)):
            print(f"PC{i+1}:", self.explained_variance_ratio[i])
    
# import pandas as pd
# df = pd.read_csv("dataset/SleepyDriverEEGBrainwave.csv")
# y = df["classification"].values
# X = df.drop("classification", axis=1)

# # Project the data onto the 2 primary principal components
# pca = PCA(3)
# pca.fit(X)
# X_projected = pca.transform(X)
# pca.show_explained_variance_ratio()

# print("Shape of X:", X.shape)
# print("Shape of transformed X:", X_projected.shape)
# print(pd.DataFrame(X_projected).head())

# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]
# x3 = X_projected[:, 2]

# import matplotlib.pyplot as plt
# from matplotlib import colormaps
# from mpl_toolkits.mplot3d import Axes3D

# # plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=colormaps.get_cmap("viridis"))

# # plt.xlabel("Principal Component 1")
# # plt.ylabel("Principal Component 2")
# # plt.colorbar()
# # plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Make sure you have data for x1, x2, x3 and color values (y)
# plt.figure(figsize=(8, 6))  # Adjust figure size as needed
# ax = plt.axes(projection='3d')

# # Plot the scatter points with colormap
# ax.scatter(x1, x2, x3, c=y, cmap=colormaps.get_cmap("viridis"), alpha=0.8, edgecolor='none')

# # Add labels for each axis
# ax.set_xlabel("Principal Component 1")
# ax.set_ylabel("Principal Component 2")
# ax.set_zlabel("Principal Component 3")

# # Add a colorbar
# # plt.colorbar(label='Color Label')  # Replace 'Color Label' with a meaningful label

# # Set viewing angles (optional)
# ax.view_init(elev=15, azim=-60)  # Adjust elevation and azimuth for a better view

# plt.title("3D Scatter Plot of Principal Components")  # Add a title (optional)

# plt.tight_layout()
# plt.show()
