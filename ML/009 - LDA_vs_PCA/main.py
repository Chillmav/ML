import numpy as np
import matplotlib.pyplot as plt



np.random.seed(42)
n = 100
data = []


for _ in range(n):
    label = np.random.choice([0, 1])

    # ogromna wariancja w osi X (szum)
    x = np.random.normal(0, 15)

    if label == 0:
        y = np.random.normal(-2, 0.5)
    else:
        y = np.random.normal(2, 0.5)

    data.append([x, y, label])

data = np.array(data)

class LDA:

    def __init__(self, k, c):
        self.k = k
        self.c = c

    def fit(self, data):

        S_w, means, occurences = self.within_class_scatter_matrix(data=data)
        S_b = self.between_class_scatter_matrix(data, means, occurences)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_w) @ S_b)

        print(eigenvalues)
        print(eigenvectors)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvectors = eigenvectors[:, :self.k]

        lda = np.dot(eigenvectors.T, data[:, :2].T)

        lda = np.c_[lda.T, data[:, 2:]]
        return lda

    def within_class_scatter_matrix(self, data):
        X = data[:, :2] - np.mean(data[:, :2], axis=0)
        data = np.c_[X, data[:, 2:]]
        S_w = np.array(
            [[0.0, 0.0],
            [0.0, 0.0]]
            )
        means = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        occurrences_arr = np.array([0, 0, 0])
        for c in range(self.c):

            mean_vector = np.array([0.0, 0.0])
            occurrences = 0
            for i in range(len(data)):
                if int(data[i, 2]) == c:
                    mean_vector[0] += data[i, 0]
                    mean_vector[1] += data[i, 1]
                    occurrences += 1
            mean_vector /= occurrences
            means[c] = mean_vector
            occurrences_arr[c] = occurrences
            for i in range(len(data)):
                if int(data[i, 2]) == c:
                    S_w += (np.array([[data[i, 0]], [data[i , 1]]]) - mean_vector) @ (np.array([[data[i, 0]], [data[i , 1]]]) - mean_vector).T


        return S_w, means, occurrences_arr

    def between_class_scatter_matrix(self, data, means, occurences):

        S_b = np.array(
            [[0.0, 0.0],
            [0.0, 0.0]]
            )
        mean = np.mean(data[:, :2], axis=0)
        for c in range(self.c):
            S_b += occurences[c] * ((np.array([[means[c, 0]], [means[c, 1]]]) - np.array([[mean[0]], [mean[1]]])) @ (np.array([[means[c, 0]], [means[c, 1]]]) - np.array([[mean[0]], [mean[1]]])).T)
        
        return S_b

class PCA:

    def __init__(self, dim):
        self.dim = dim

    def fit(self, data):

        X = data[:, :2]
        X_centered = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        cov = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        idx = np.argsort(eigenvalues)[::-1] # decreasing order

        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :self.dim]

        projections = np.dot(X_centered, eigenvectors)
        
        return eigenvectors.T, eigenvalues[:self.dim], projections
        


def main(data):

    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    colors = ["blue", "green", "orange"]

    for point in data:
        axes[0].scatter(point[0], point[1], color=colors[int(point[2])])

    axes[0].set_title("Raw Data")
    axes[0].axhline(0)
    axes[0].axvline(0)

    pca = PCA(dim=1)
    _, _, projections = pca.fit(data)

    for i, point in enumerate(data):
        axes[1].scatter(projections[i], 0, color=colors[int(point[2])])

    axes[1].set_title("PCA 1D")
    axes[1].axhline(0)

    lda = LDA(k=1, c=2)
    projections = lda.fit(data)

    for i in range(len(projections)):

        axes[2].scatter(projections[i, 0], 0, color=colors[int(projections[i, 1])])
        axes[2].set_title("LDA 1D")
        axes[2].axhline(0)

    pca = PCA(dim=2)
    _, _, projections = pca.fit(data)

    for i, point in enumerate(data):
        axes[3].scatter(projections[i, 0], projections[i, 1], color=colors[int(point[2])])

    axes[3].set_title("PCA 2D")
    axes[3].axhline(0)

    lda = LDA(k=2, c=2)
    projections = lda.fit(data)

    for i in range(len(projections)):

        axes[4].scatter(projections[i, 0], projections[i, 1], color=colors[int(projections[i, 2])])
        axes[4].set_title("LDA 2D")
        axes[4].axhline(0)


    plt.tight_layout()
    plt.show()

main(data)
