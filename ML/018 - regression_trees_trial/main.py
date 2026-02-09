import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

data_size = 100

X = np.linspace(0, 10, data_size)
X = X.reshape(-1, 1)

Y = np.where(
    X[:, 0] < 3,
    X[:, 0] ** 2,
    np.where(
        X[:, 0] < 7,
        10 + 0.5 * X[:, 0],
        25 - (X[:, 0] - 7) ** 2
    )
)

Y += np.random.normal(0, 0.5, size=data_size)

class Node:

    def __init__(self, feature: int = None, threshold: float = None, right = None, left = None, value: int = None):

        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value
        
class RegressionTree:

    def __init__(self, X: np.ndarray, Y: np.ndarray, min_obs=10):

        self.root = None
        self.k = min_obs
        self.X = X
        self.Y = Y
        self.leafs = 0
        self.alphas = []

    def build(self, X, Y):

        if (Y.shape[0] <= self.k and Y.shape[0] > 0):
            self.leafs += 1
            return Node(value=np.mean(Y)) # LEAF
        if (len(Y) == 0):
            return None
        
        feature, threshold = self.branch(X, Y)
        X_L, Y_L, X_R, Y_R = self.split_dataset(X, Y, feature=feature, threshold=threshold)
        node = Node(feature, threshold)
        node.right = self.build(X_R, Y_R)
        node.left = self.build(X_L, Y_L)

        return node
    
    def branch(self, X: np.ndarray, Y: np.ndarray):

        data_dim = X.shape[1]
        results = np.zeros((X.shape[1], 3))
        for j in range(data_dim):

            X_sorted = np.sort(X[:, j])
            thresholds = np.zeros(X.shape[0] - 1)

            for i in range(X.shape[0] - 1):

                thresholds[i] = (X_sorted[i] + X_sorted[i + 1]) / 2
            result = [j, thresholds[0], np.inf]

            for threshold in thresholds:

                _, Y_L, _, Y_R = self.split_dataset(X, Y, j, threshold)
                if len(Y_L) == 0 or len(Y_R) == 0:
                    continue

                SSR = self.SSR(np.mean(Y_L), Y_L) + self.SSR(np.mean(Y_R), Y_R)

                if (result[2] > SSR):

                    result[1] = threshold
                    result[2] = SSR

            results[j] = result

        best = results[np.argmin(results[:, 2])]

        return int(best[0]), best[1]

    def split_dataset(self, X, Y, feature:int, threshold:float):

        mask = X[:, feature] < threshold

        return X[mask], Y[mask], X[~mask], Y[~mask]

    def cross_validation(self, X, Y):

        data = np.c_[X, Y]
        size = data.shape[0]
        train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
        test_data = np.delete(data, train_data_indices, axis=0)
        train_data = data[train_data_indices]

        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]
    
    def predict(self, node: Node, X, Y):

        if (node.value != None):

            for i in range(Y.shape[0]):
                print(f"Real: {Y[i]} | Predicted: {node.value}")
                plt.scatter(X[i, 0], node.value, color="green")
            return self.SSR(node.value, Y)
        
        feature = node.feature
        threshold = node.threshold
        X_left, Y_left, X_right, Y_right = self.split_dataset(X, Y, feature=feature, threshold=threshold)

        return (

            self.predict(node.right, X_right, Y_right) + 
            self.predict(node.left, X_left, Y_left) 

        )
    
    def SSR(self, mean, Y):

        return np.sum((mean - Y)**2)

    def calc_tree_score(self, node: Node, X: np.ndarray, Y: np.ndarray):

        if (node.value != None):
            return self.SSR(node.value, Y)
        
        feature = node.feature
        threshold = node.threshold
        X_left, Y_left, X_right, Y_right = self.split_dataset(X, Y, feature=feature, threshold=threshold)

        return (

            self.calc_tree_score(node.right, X_right, Y_right) + 
            self.calc_tree_score(node.left, X_left, Y_left) 

        )

    def prune(self, node: Node, X: np.ndarray, Y: np.ndarray, fold=10):

        averages = np.zeros(self.leafs)

        for i in range(fold):

            X_train, Y_train, X_test, Y_test = self.cross_validation(X, Y)
            while (self.leafs):
                pass

    def calc_alphas(self, node: np.ndarray, X: np.ndarray, Y: np.ndarray):

        leafs = self.leafs
        self.alphas = np.zeros(self.leafs) # from deepest
        tree_score = self.calc_tree_score(node, X, Y) + leafs * self.alphas[0]

        for i in range(self.leafs):
            leafs -= 1
            new_tree_score = np.inf
            while new_tree_score > tree_score:
                self.alphas[i+1] += 1
                self.calc_tree_score(node, X, Y)



model = RegressionTree(X, Y, 10)
model.root = model.build(X, Y)
X_train, Y_train, X_test, Y_test = model.cross_validation(X, Y)
SSE = model.predict(model.root, X_test, Y_test)
print(f"SSE: {SSE}")
model.calc_alphas(model.root, X, Y)

plt.scatter(X, Y, color="blue")
plt.show()


