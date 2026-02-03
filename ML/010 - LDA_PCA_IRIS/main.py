import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PCA:

    def __init__(self, k):

        self.k = k

    def train(self, standard_data):

        cov_matrix = np.cov(standard_data, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvectors = eigenvectors[:, :self.k]

        pca = standard_data @ eigenvectors

        varations = eigenvalues / np.sum(eigenvalues)

        return varations, eigenvectors, pca

    def visualize(self, standard_data, y):

        x = standard_data
        variations, eigenvectors, pca = self.train(standard_data=x)
        
        figure, axes = plt.subplots(1, 3, figsize=(10, 5)) # loadings, scree_plot, pca_2D
        
        # loadings
        max_val = np.max(np.abs(eigenvectors[:, :2]))

        labels = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
        pc1_x = eigenvectors[:, 0]
        pc1_y = eigenvectors[:, 1]

        axes[0].scatter(eigenvectors[:, 0], eigenvectors[:, 1])
        for (i, label) in enumerate(labels):
            axes[0].text(pc1_x[i], pc1_y[i], label)

        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_xlim(-max_val, max_val)
        axes[0].set_ylim(-max_val, max_val)

        axes[0].set_title("PCA Loadings")

        # Scree plot
        labels = [f"PC-{i + 1}" for i in range(len(variations))]
        axes[1].bar(labels, variations)
        axes[1].set_xlabel("Principal Components")
        axes[1].set_ylabel("Explained Variance")
        axes[1].set_title("Explained Variance per Principal Component")

        # pca_2d
        pc1 = pca[:, 0]
        pc2 = pca[:, 1]
        colors = ["purple", "pink", "blue"]

        for index, y_i in enumerate(y):
            axes[2].scatter(pc1[index], pc2[index], color=colors[int(y_i)])

        axes[2].set_xlabel("PC1")
        axes[2].set_ylabel("PC2")
        axes[2].set_title("PCA")

        plt.tight_layout()

class LDA:

    def __init__(self, k):

        self.k = k

    def train(self, x):

        iris_setosa = x[0:50]
        iris_versicolor = x[50:100]
        iris_virginica = x[100:150]

        setosa_mean = np.mean(iris_setosa, axis=0)
        versicolor_mean = np.mean(iris_versicolor, axis=0)
        virginica_mean = np.mean(iris_virginica, axis=0)
        mean_all = np.mean(x, axis=0)
        means = [setosa_mean, versicolor_mean, virginica_mean]

        s_c_setosa = (iris_setosa - setosa_mean).T @ (iris_setosa - setosa_mean)
        s_c_versicolor = (iris_versicolor - versicolor_mean).T @ (iris_versicolor - versicolor_mean)
        s_c_virginica = (iris_virginica - virginica_mean).T @ (iris_virginica - virginica_mean)

        S_w = s_c_setosa + s_c_virginica + s_c_versicolor

        S_b = np.zeros((4, 4))

        for mean in means:
            diff = (mean - mean_all).reshape(-1, 1)
            S_b += 50 * (diff @ diff.T)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_w) @ S_b)


        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvectors = eigenvectors[:, :self.k]

        variances = eigenvalues / np.sum(eigenvalues)

        return variances, eigenvectors, x @ eigenvectors

    def visualize(self, x, y):

        variations, eigenvectors, lda = self.train(x)
        
        figure, axes = plt.subplots(1, 3, figsize=(10, 5)) # loadings, scree_plot, pca_2D
        
        # loadings
        max_val = np.max(np.abs(eigenvectors[:, :2]))

        labels = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
        ld1_x = eigenvectors[:, 0]
        ld1_y = eigenvectors[:, 1]

        axes[0].scatter(eigenvectors[:, 0], eigenvectors[:, 1])
        for (i, label) in enumerate(labels):
            axes[0].text(ld1_x[i], ld1_y[i], label)

        axes[0].set_xlabel("LD1")
        axes[0].set_ylabel("LD2")
        axes[0].set_xlim(-max_val, max_val)
        axes[0].set_ylim(-max_val, max_val)

        axes[0].set_title("Discriminant vectors")

        # Scree plot
        labels = [f"LD-{i + 1}" for i in range(len(variations))]
        axes[1].bar(labels, variations)
        axes[1].set_xlabel("LDA")
        axes[1].set_ylabel("Contribution to distinguishing")
        axes[1].set_title("Explained contribution per LD")

        # lda_2d
        ld1 = lda[:, 0]
        ld2 = lda[:, 1]
        colors = ["purple", "pink", "blue"]

        for index, y_i in enumerate(y):
            axes[2].scatter(ld1[index], ld2[index], color=colors[int(y_i)])

        axes[2].set_xlabel("LD1")
        axes[2].set_ylabel("LD2")
        axes[2].set_title("LDA")

        plt.tight_layout()

def main(k):

    df = pd.read_csv("ML/010-LDA_PCA_IRIS/Iris.csv", delimiter=',')
    X = df[[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]].values
    x = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = df["Species"].map({
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }).values
    
    global_variance = np.var(X, axis=0, ddof=0)
    print(global_variance)
    pca = PCA(2)
    pca.visualize(x, y)
    lda = LDA(2)
    lda.visualize(x, y)
    plt.show()

    



main(2)