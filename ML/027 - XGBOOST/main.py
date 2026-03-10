import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
diabetes = load_diabetes()
X = diabetes.data
Y = diabetes.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, gain=None, weight=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.value = value
        self.gain = gain
        self.weight = weight

class XGBoost:

    def __init__(self, lr=.3, trees=100, gamma=2500, max_depth=6, alpha=20, min_samples_split = 5):

        self.scaler = StandardScaler()
        self.lr = lr
        self.trees = trees
        self.gamma = gamma
        self.max_depth = max_depth
        self.alpha = alpha
        self.min_samples_split = min_samples_split
        self.forest = []
        self.total_prunes = 0

    def fit(self, X, Y):

        X = self.scaler.fit_transform(X)
        self.build_forest(X, Y)

    def update_residuals(self, residuals, outcomes):

        return residuals - (self.lr * outcomes)

    def build_forest(self, X, Y):

        self.init_pred = np.mean(Y)
        residuals = Y - self.init_pred

        for _ in range(self.trees):

            root = self.build_tree(X, residuals)
            self.prune(root)
            self.forest.append(root)
            outcome = np.zeros(len(Y))
            for i in range(len(Y)):
                outcome[i] = self.compute_outcome(root, X[i])
            residuals = self.update_residuals(residuals, outcome)


    def calc_output(self, Y):

        return np.sum(Y) / (len(Y) + self.alpha)

    def build_tree(self, X, Y, level=0):

        n_features = X.shape[1]

        if level >= self.max_depth:
            node = Node(value=self.calc_output(Y), weight=len(Y))

            return node
        
        if len(Y) < self.min_samples_split:
            node = Node(value=self.calc_output(Y), weight=len(Y))

            return node
        
        best_feature = 0
        best_threshold = 0
        best_gain = 0

        for feature in range(n_features):
            
            x = X[:, feature]
            x_sorted = np.sort(x)
            thresholds = np.unique((x_sorted[:-1] + x_sorted[1:]) / 2)

            for threshold in thresholds:

                X_left, Y_left, X_right, Y_right = self.split_dataset(x, Y, threshold)
                
                if len(Y_left) == 0 or len(Y_right) == 0:
                    continue

                gain = self.similarity_score(Y_left) + self.similarity_score(Y_right) - self.similarity_score(np.concatenate((Y_left, Y_right)))
                
                if gain > best_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_gain = gain

        if best_gain <= 0:
            return Node(value=self.calc_output(Y), weight=len(Y))
        
        X_left, Y_left, X_right, Y_right = self.split_whole_dataset(X, Y, best_feature, best_threshold)
        node = Node(best_feature, best_threshold)
        node.gain = best_gain
        node.left = self.build_tree(X_left, Y_left, level + 1)
        node.right = self.build_tree(X_right, Y_right, level + 1)
        return node
    
    def prune(self, root: Node):

        while True:
            
            if root.value is not None:
                return root
            
            node, _ = self.deepest_node(root)

            if node is None:
                break

            if (node.gain < self.gamma):

                self.total_prunes += 1
                w1, v1 = node.right.weight, node.right.value
                w2, v2 = node.left.weight, node.left.value
                w1 = 0 if w1 == None else w1
                w2 = 0 if w2 == None else w2
                v1 = 0 if v1 == None else v1
                v2 = 0 if v2 == None else v2
                node.value = (w1 * v1 + w2 * v2) / (w1 + w2 + self.alpha)
                node.right = None
                node.left = None

            else:
                break
    
    def deepest_node(self, node: Node, depth=0):

        if node is None or node.value is not None:
            return None, -1

        if (
            node.left and node.right
            and node.left.value is not None
            and node.right.value is not None
        ):
            return node, depth

        node_left, depth_left = self.deepest_node(node.left, depth + 1)
        node_right, depth_right = self.deepest_node(node.right, depth + 1)

        if depth_left > depth_right:
            return node_left, depth_left
        else:
            return node_right, depth_right

    def similarity_score(self, residuals):

        return (np.sum(residuals) ** 2) / (len(residuals) + self.alpha)
    
    def split_whole_dataset(self, X, Y, feature: int, threshold:float):

        mask = X[:, feature] < threshold

        return X[mask], Y[mask], X[~mask], Y[~mask]
    
    def split_dataset(self, X, Y, threshold:float):

        mask = X < threshold

        return X[mask], Y[mask], X[~mask], Y[~mask]
    
    def compute_outcome(self, root: Node, x: np.ndarray):

        if root.value is not None:
            return root.value
        
        if root.right == None:
            return 0
        if root.left == None:
            return 0
        
        if x[root.feature] < root.threshold:
            return self.compute_outcome(root.left, x)
        else:
            return self.compute_outcome(root.right, x)
        
    def predict(self, X):

        X = self.scaler.transform(X)
        n = X.shape[0]

        predictions = np.full(n, self.init_pred)

        for tree in self.forest:
            predictions += self.lr * np.array(
                [self.compute_outcome(tree, x) for x in X]
            )

        return predictions

    def evaluate(self, Y_predicted, Y_real):

            for y_hat, y in zip(Y_predicted, Y_real):
                print(f"Real: {y} | Predicted: {y_hat}")

            print(f"SSE: {self.SSE(Y_predicted, Y_real)}")
    
    def SSE(self, Y_predicted, Y_real):

        return np.sum((Y_predicted - Y_real)**2)

model = XGBoost()
model.fit(X_train, Y_train)
print(model.total_prunes)
predictions = model.predict(X_test)
model.evaluate(predictions, Y_test)