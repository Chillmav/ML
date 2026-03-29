import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv("PCA_LDA_project\data\Iris.csv")

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
y = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).values


fig, axis = plt.subplots(1, 3, figsize=(10, 5))

class LDA:
    
    def __init__(self):
        
        self.n_labels = 0
        self.n_features = 0
        self.means = 0
        self.k = 0
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray, k=1):
        
        self.k = k
        S_within, S_between = self.calc_matrices(X, y)
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_within) @ S_between)
        indices = np.argsort(eigenvalues)[::-1][:k]
        best_eigenvalues = eigenvalues[indices]
        best_eigenvectors = eigenvectors[:, indices]
        X_new = X @ best_eigenvectors
        self.visualize(X_new, y)
            
    def visualize(self, X_new, y):
        
        colors = ["blue", "orange", "green"]
        
        if self.k == 1:
            
            for label in np.unique(y):
                axis[0].scatter(
                    X_new[y == label, 0],
                    np.zeros_like(X_new[y == label, 0]),
                    color=colors[label],
                    label=f"class {label}"
                )
            
            axis[0].set_title("LDA_scratch (1D)")
            axis[0].set_xlabel("LD1")
            axis[0].set_ylabel("LD2")
            axis[0].legend()
        
        if self.k == 2:
            
            for label in np.unique(y):
                
                axis[1].scatter(
                    X_new[y == label, 0],
                    X_new[y == label, 1],
                    color=colors[label],
                    label=f"class {label}"
                )
            
            axis[1].set_title("LDA_scratch (2D)")
            axis[1].set_xlabel("LD1")
            axis[1].set_ylabel("LD2")
            axis[1].legend()
            
    def calc_matrices(self, X, y):
                
        self.n_features = X.shape[1]
        self.n_labels = len(np.unique(y))
        self.means = np.zeros((self.n_labels, self.n_features))
        labels, counts = np.unique(y, return_counts=True)
        
        S_within = 0
        S_between = 0
        overall_mean = np.mean(X, axis=0)
        
        for i in range(self.n_labels):
            
            mask = y == labels[i]
            x = X[mask]    
            self.means[i] = np.mean(x, axis=0)
            S_within += (x - self.means[i]).T @ (x - self.means[i])
            S_between += counts[i] * ((self.means[i] - overall_mean).reshape(-1, 1) @ (self.means[i] - overall_mean).reshape(-1, 1).T)
            
        return S_within, S_between
    
model1D =  LDA()
model2D = LDA()
model_sklearn = LinearDiscriminantAnalysis(n_components=2)

X_lda = model_sklearn.fit_transform(X, y)
variance = model_sklearn.explained_variance_ratio_
loadings = model_sklearn.scalings_

model1D.fit_transform(X, y, k=1)
model2D.fit_transform(X, y, k=2)

for label in np.unique(y):
    
    axis[2].scatter(X_lda[y == label, 0],
                X_lda[y == label, 1],
                label=f"class {label}")

axis[2].set_title("LDA_sklearn (2D)")
axis[2].set_xlabel("LD1")
axis[2].set_ylabel("LD2")
axis[2].legend()

feature_names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
for i in range(2):
    print(f"\n[sklearn] LD{i+1} loadings:")
    for f, val in zip(feature_names, loadings[:, i]):
        print(f"{f}: {val:.4f}")        
        
plt.show()

# variance [0.99147248 0.00852752]

# 🎯 LD1 (first discriminant)

# Loadings:

# SepalLength: +0.82
# SepalWidth: +1.55
# PetalLength: −2.18
# PetalWidth: −2.85
# 👉 Meaning
# Dominated by petal features (largest magnitudes)
# Opposes:
# large petals (negative side)
# vs small petals + wider sepals (positive side)
# 👉 Interpretation

# LD1 separates classes mainly by petal size

# ✔ This is the main separation axis (≈99% variance)