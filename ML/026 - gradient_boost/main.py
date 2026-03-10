import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

diabetes = datasets.load_diabetes()
X_train, X_test, Y_train, Y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.20)

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.value = value

class GradientBoostRegressionTrees:

    def __init__(self, trees = 200, max_depth = 3, lr=0.05, min_samples_split = 5):

        self.forest = []
        self.scalar = StandardScaler()
        self.M = trees
        self.lr = lr
        self.md = max_depth
        self.min_samples_split = min_samples_split
        self.init_pred = 0

    def train(self, X, Y):

        X = self.scalar.fit_transform(X)
        self.build_forest(X, Y)

    def predict(self, X):

        X = self.scalar.transform(X)
        n = X.shape[0]

        predictions = np.full(n, self.init_pred)

        for tree in self.forest:
            for i in range(n):
                predictions[i] += self.lr * self.compute_outcome(tree, X[i])

        return predictions
    
    def build_forest(self, X, Y):

        self.init_pred = np.mean(Y)
        residuals = Y - self.init_pred

        for _ in range(self.M):

            root = self.build_tree(X, residuals)
            self.forest.append(root)
            outcome = np.zeros(len(Y))
            for i in range(len(Y)):
                outcome[i] = self.compute_outcome(root, X[i])
            residuals = self.update_residuals(residuals, outcome)

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
    
    def build_tree(self, X, Y, level=0):

        n_features = X.shape[1]

        if level >= self.md:
            return Node(value=np.mean(Y))
        
        if len(Y) < self.min_samples_split:
            return Node(value=np.mean(Y))
        
        best_feature = 0
        best_threshold = 0
        best_ssr = 1e9

        for feature in range(n_features):
            
            x = X[:, feature]
            x_sorted = np.sort(x)
            thresholds = np.unique((x_sorted[:-1] + x_sorted[1:]) / 2)

            for threshold in thresholds:

                ssr = 0
                X_left, Y_left, X_right, Y_right = self.split_dataset(x, Y, threshold)
                
                if len(Y_left) == 0 or len(Y_right) == 0:
                    continue

                ssr = self.SSR(np.mean(Y_left), Y_left) + self.SSR(np.mean(Y_right), Y_right)

                if ssr < best_ssr:
                    best_feature = feature
                    best_threshold = threshold
                    best_ssr = ssr

        if best_ssr == 1e9:
            return Node(value=np.mean(Y))
        
        X_left, Y_left, X_right, Y_right = self.split_whole_dataset(X, Y, best_feature, best_threshold)
        node = Node(best_feature, best_threshold)
        node.left = self.build_tree(X_left, Y_left, level + 1)
        node.right = self.build_tree(X_right, Y_right, level + 1)
        return node
    
    def SSR(self, mean, Y):

        return np.sum((Y - mean)**2)
    
    def update_residuals(self, residuals, outcomes):

        return residuals - (self.lr * outcomes)
    
    def split_whole_dataset(self, X, Y, feature: int, threshold:float):

        mask = X[:, feature] < threshold

        return X[mask], Y[mask], X[~mask], Y[~mask]
    
    def split_dataset(self, X, Y, threshold:float):

        mask = X < threshold

        return X[mask], Y[mask], X[~mask], Y[~mask]
    
    def evaluate(self, Y_predicted, Y_real):

        for y_hat, y in zip(Y_predicted, Y_real):
            print(f"Real: {y} | Predicted: {y_hat}")

        print(f"SSE: {self.SSE(Y_predicted, Y_real)}")
    
    def SSE(self, Y_predicted, Y_real):

        return np.sum((Y_predicted - Y_real)**2)

model = GradientBoostRegressionTrees()
model.train(X_train, Y_train)
predictions = model.predict(X_test)
model.evaluate(predictions, Y_test)