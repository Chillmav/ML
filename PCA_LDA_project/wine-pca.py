import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

data = load_wine()
X = data.data
Y = data.target
feature_names = data.feature_names

class PCA_scratch:

    def __init__(self, k=2):

        self.k = k
        self.variance = []

    def fit_transform(self, X):

        n_samples = X.shape[0]
        X_centered = (X - np.mean(X, axis = 0)) / np.std(X, axis=0)
        cov = X_centered.T @ X_centered / (n_samples - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        self.calc_variance_ratio(eigenvalues)
        indices = np.argsort(eigenvalues)[::-1][:self.k]

        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        projections = X_centered @ eigenvectors

        return eigenvectors, eigenvalues, projections
    
    def calc_variance_ratio(self, eigenvalues):

        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]

        sum = np.sum(eigenvalues)
        self.variance = [value / sum for value in eigenvalues][:self.k]


pca_from_scratch = PCA_scratch(k=3)
sklearn_model = PCA(n_components = 3, svd_solver="full")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_sklearn = sklearn_model.fit_transform(X_scaled)
variance_ratio = sklearn_model.explained_variance_ratio_
eigenvectors, eigenvalues, transformed_X  = pca_from_scratch.fit_transform(X)
scratch_variance_ratio = pca_from_scratch.variance

sklearn_components = sklearn_model.components_
print(sklearn_components)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

colors = ["red", "green", "blue"]
labels = ["Class 1", "Class 2", "Class 3"]

for i in range(3):
    ax1.scatter(
        X_sklearn[Y == i, 0],
        X_sklearn[Y == i, 1],
        X_sklearn[Y == i, 2],
        color=colors[i],
        label=labels[i],
        alpha=0.7
    )

ax1.set_xlabel(f"PC1 ({np.round(variance_ratio[0]*100)}%)")
ax1.set_ylabel(f"PC2 ({np.round(variance_ratio[1]*100)}%)")
ax1.set_zlabel(f"PC3 ({np.round(variance_ratio[2]*100)}%)")
ax1.set_title("Sklearn PCA")
ax1.legend()


for i in range(3):
    ax2.scatter(
        transformed_X[Y == i, 0],
        transformed_X[Y == i, 1],
        transformed_X[Y == i, 2],
        color=colors[i],
        label=labels[i],
        alpha=0.7
    )

ax2.set_xlabel(f"PC1 ({np.round(scratch_variance_ratio[0]*100)}%)")
ax2.set_ylabel(f"PC2 ({np.round(scratch_variance_ratio[1]*100)}%)")
ax2.set_zlabel(f"PC3 ({np.round(scratch_variance_ratio[2]*100)}%)")
ax2.set_title("PCA from scratch")
ax2.legend()

ax1.view_init(elev=20, azim=45)
ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()


