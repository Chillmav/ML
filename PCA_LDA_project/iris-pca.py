import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

df = pd.read_csv("PCA_LDA_project\data\Iris.csv")

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
Y = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).values


class PCA_scratch:

    def __init__(self, k=2):

        self.k = k
        self.variance = []

    def fit_transform(self, X):

        n_samples = X.shape[0]
        X_centered = (X - np.mean(X, axis = 0))
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


pca_from_scratch = PCA_scratch(k=2)
sklearn_model = PCA(n_components = 2, svd_solver="full")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_sklearn = sklearn_model.fit_transform(X_scaled)
variance_ratio = sklearn_model.explained_variance_ratio_
eigenvectors, eigenvalues, transformed_X  = pca_from_scratch.fit_transform(X)
scratch_variance_ratio = pca_from_scratch.variance

sklearn_components = sklearn_model.components_
# print(sklearn_components)

fig, axis = plt.subplots(1, 2, figsize=(10, 5))

colors = ["red", "green", "blue"]
labels = ["Setosa", "Versicolor", "Virginica"]

for i in range(3):
    axis[0].scatter(
        X_sklearn[Y == i, 0],
        X_sklearn[Y == i, 1],
        color=colors[i],
        label=labels[i]
    )

axis[0].set_xlabel(f"PC1 - {np.round(variance_ratio[0] * 100)}%")
axis[0].set_ylabel(f"PC2 - {np.round(variance_ratio[1] * 100)}%")
axis[0].set_title("Sklearn PCA")
axis[0].legend()


for i in range(3):
    axis[1].scatter(
        transformed_X[Y == i, 0],
        transformed_X[Y == i, 1],
        color=colors[i],
        label=labels[i]
    )

axis[1].set_xlabel(f"PC1 - {np.round(scratch_variance_ratio[0] * 100)}%")
axis[1].set_ylabel(f"PC2 - {np.round(scratch_variance_ratio[1] * 100)}%")
axis[1].set_title("PCA from scratch")
axis[1].legend()

plt.show()

# Iris
# PC1 = 0.361 * SepalLengthCm - 0.08 * SepalWidthCm + 0.85 * PetalLengthCm + 0.36 * PetalWidthCm
# PC2 = 0.656 * SepalLengthCm + 0.73 * SepalWidthCm - 0.17 * PetalLengthCm - 0.07 * PetalWidthCm



