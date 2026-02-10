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

    def split(self, X, Y):

        data = np.c_[X, Y]
        size = data.shape[0]
        train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
        test_data = np.delete(data, train_data_indices, axis=0)
        train_data = data[train_data_indices]

        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]
    
    def predict(self, node: Node, X, Y):

        if (node.value != None):

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

    def SSE_node(self, node: Node, X, Y):
                
        if (node.value != None):
            return self.SSR(node.value, Y)

        return self.SSR(np.mean(Y), Y)

    def calc_leafs(self, node: Node):

        if (node.value != None):
            return 1

        return (
            self.calc_leafs(node.right) + self.calc_leafs(node.left)
        )
    
    def calc_alpha(self, node: Node, X, Y):

        SSE_subtree = self.calc_tree_score(node, X, Y)
        SSE_node = self.SSE_node(node, X, Y)
        leafs = self.calc_leafs(node)
        if (leafs == 1):
            return np.inf
        else:
            return (SSE_node - SSE_subtree) / (leafs - 1)

    def find_weakest(self, node: Node, X, Y):

        if (node.value != None):
            return np.inf, None, None, None
    
        alpha1 = self.calc_alpha(node, X, Y)
        feature = node.feature
        threshold = node.threshold
        X_left, Y_left, X_right, Y_right = self.split_dataset(X, Y, feature=feature, threshold=threshold)

        alpha2, node2, _, _ = self.find_weakest(node.right, X_right, Y_right)
        alpha3, node3, _, _ = self.find_weakest(node.left, X_left, Y_left)

        choices = np.array([alpha1, alpha2, alpha3])
        nodes = [node, node2, node3]
        data = [[X, Y], [X_right, Y_right], [X_left, Y_left]]
        idx = np.argmin(choices)
        return choices[idx], nodes[idx], data[idx][0], data[idx][1]
    
    def prune(self, node: Node, X, Y):

        right_leafs = self.calc_leafs(node.right)
        left_leafs = self.calc_leafs(node.left)
        node.right = None
        node.left = None
        node.value = np.mean(Y)
        self.leafs -= (right_leafs + left_leafs - 1)
    
    def build_prune(self, root: Node, X, Y):

        results = np.zeros((self.leafs - 1, 4))
        i = 0
        while self.leafs > 1:
            alpha, node, x, y = self.find_weakest(root, X, Y)
            self.prune(node, x, y)
            results[i, 0] = alpha
            results[i, 1] = self.calc_tree_score(root, X, Y)
            results[i, 2] = self.calc_leafs(root)
            i += 1
        return results
    
    def CV(self, X, Y, folds=10):

        model = RegressionTree(X, Y)
        model.root = model.build(X, Y)
        pruning_path = model.build_prune(model.root, X, Y)
        means = np.zeros(pruning_path.shape[0])

        for _ in range(folds):
            self.leafs = 0
            X_train, Y_train, X_test, Y_test = self.split(X, Y)
            train_tree = self.build(X_train, Y_train)

            means[0] += self.predict(train_tree, X_test, Y_test)

            for j, alpha in enumerate(pruning_path):
                _, node, X_node, Y_node = self.find_weakest(train_tree, X_train, Y_train)
                if node is None:
                    break 
                self.prune(node, X_node, Y_node)
                means[j+1] += self.predict(train_tree, X_test, Y_test)

        means /= folds
        return means
        
model = RegressionTree(X, Y, 10)
means = model.CV(X, Y, 10)
print(means)
plt.plot(means)
plt.scatter(X, Y, color="blue")
plt.show()


