import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Simple matrix

A = np.array([[3, 1], [1, 3], [1, -1]])

# Iris Dataset

df = pd.read_csv("ML/data/Iris.csv")
X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values

def SVD(A: np.ndarray):

    ATA = A.T @ A
    eigvals, V = np.linalg.eigh(ATA)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    singular_values = np.sqrt(eigvals)
    sigma = np.diag(singular_values)
    V = V[:, idx]
    U = A @ V @ np.linalg.inv(sigma)
    return U, sigma, V.T

def PCA(X: np.ndarray, k=2):

    x_mean = np.mean(X, axis=0).reshape(-1, 1).T
    ones = np.ones(X.shape[0]).reshape(-1, 1)
    X_mean = ones @ x_mean

    B = X - X_mean # data with zero mean (in theory)

    U, sigma, V_T = SVD(B)
    singular_values = np.diag(sigma)
    var_contribution = singular_values**2 / np.sum(singular_values**2)
    print(var_contribution)
    sigma = sigma[:, :k]
    PCA = U @ sigma
    print(PCA)

    plt.scatter(PCA[:, 0], PCA[:, 1])
    for i in range(k):
        print(f"{i}PC covers {int(np.round(np.sum(var_contribution[i]), 2) * 100)}% variance")
    plt.show()

    
PCA(X)