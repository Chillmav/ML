import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = data.data
y = data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# ===== LDA SCRATCH =====
# ======================

class LDA:
    
    def fit_transform(self, X, y, k=2):
        
        S_w, S_b = self.calc_matrices(X, y)
        
        A = np.linalg.pinv(S_w) @ S_b
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        indices = np.argsort(eigenvalues)[::-1][:k]
        W = eigenvectors[:, indices]
        
        X_new = (X @ W).real
        return X_new
    
    def calc_matrices(self, X, y):
        
        n_features = X.shape[1]
        labels, counts = np.unique(y, return_counts=True)
        
        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))
        
        overall_mean = np.mean(X, axis=0)
        
        for i in range(len(labels)):
            
            x = X[y == labels[i]]
            mean = np.mean(x, axis=0)
            
            S_w += (x - mean).T @ (x - mean)
            
            mean_diff = (mean - overall_mean).reshape(-1, 1)
            S_b += counts[i] * (mean_diff @ mean_diff.T)
            
        return S_w, S_b


# ======================
# ===== SKLEARN ========
# ======================

lda_sklearn = LinearDiscriminantAnalysis(n_components=2)
X_sklearn = lda_sklearn.fit_transform(X_scaled, y)
variance = lda_sklearn.explained_variance_ratio_
print("\nVariance ratio:")
print(variance)

loadings = lda_sklearn.scalings_
feature_names = data.feature_names

for i in range(2):
    print(f"\n[Wine sklearn] LD{i+1} loadings:")
    for f, val in zip(feature_names, loadings[:, i]):
        print(f"{f}: {val:.4f}")

# ======================
# ===== SCRATCH ========
# ======================

lda_scratch = LDA()
X_scratch = lda_scratch.fit_transform(X_scaled, y, k=2)

# ======================
# ===== PLOT ===========
# ======================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ["red", "green", "blue"]

# --- sklearn ---
for i in range(3):
    axes[0].scatter(
        X_sklearn[y == i, 0],
        X_sklearn[y == i, 1],
        color=colors[i],
        label=f"class {i}",
        alpha=0.7
    )

axes[0].set_title("LDA (sklearn)")
axes[0].legend()

# --- scratch ---
for i in range(3):
    axes[1].scatter(
        X_scratch[y == i, 0],
        X_scratch[y == i, 1],
        color=colors[i],
        label=f"class {i}",
        alpha=0.7
    )

axes[1].set_title("LDA (scratch)")
axes[1].legend()

plt.tight_layout()
plt.show()

# Variance ratio: [0.68747889 0.31252111]
