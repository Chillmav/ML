import numpy as np
import matplotlib.pyplot as plt


def PCA(X, k):

    # Standarize the data

    X_standard = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # Compute the covariance matrix

    cov_matrix = np.cov(X_standard, rowvar=False) # because our columns are features

    # Compute the eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues and eigenvectors in descending order of eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Took the top k eigenvectors

    eigenvectors = eigenvectors[:, :k] # takes k cols

    # Transform the data by multiplaying the standarized data by our top k eigenvectors

    X_pca = np.matmul(X_standard, eigenvectors) # iloczyn skalarny (mapuje vektor x na kierunek v)
    print(X_pca)
    # Return eigenvectors, eigenvalues, transformed data

    return eigenvectors.T, eigenvalues[:k], X_pca


data = np.array([[2,6,3,1], [3,2,1,7], [3,4,3,2], [4,2,1,5], [1,2,8,5]])
pca = PCA(data, 2)

pc_1 = pca[2][:, :1]
pc_2 = pca[2][:, 1:]
plt.scatter(pc_1, pc_2, color="blue")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axhline(0)
plt.axvline(0)
plt.show()

