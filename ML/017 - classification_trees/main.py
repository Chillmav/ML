import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.read_csv("ML/data/Iris.csv")
data = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values

labels = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).values

data = np.c_[data, labels]
def split_dataset(data: np.ndarray):

    size = data.shape[0]
    train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
    test_data = np.delete(data, train_data_indices, axis=0)
    train_data = data[train_data_indices]
    return train_data, test_data

train_data, test_data = split_dataset(data)
X_train, Y_train, X_test, Y_test = train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.value = value # label if leave, value != None (leave)

class ClassificationTree:

    def __init__(self):
        
        self.n_labels = 0
        self.root = Node()

    def build(self, X: np.ndarray, Y: np.ndarray):

        results = []

        if len(np.unique(Y)) == 1:
            return Node(value=int(Y[0]))
        
        if len(Y) == 0:
            return None
        
        for j in range(X.shape[1]):
            min_impurity, branch_condition = self.branch(X[:, j], Y)
            results.append([min_impurity, branch_condition, int(j)])
        results = np.array(results)
        
        impurity, threshold, feature = results[np.argmin(results[:, 0])]
        feature = int(feature)
        X_left, Y_left, X_right, Y_right = self.split_data(X, Y, feature=feature, threshold=threshold)

        if len(Y_left) == 0 or len(Y_right) == 0:
            majority = int(np.bincount(Y.astype(int)).argmax())
            return Node(value=majority)
        
        node = Node(feature, threshold)
        node.left = self.build(X_left, Y_left)
        node.right = self.build(X_right, Y_right)
        
        return node


    def branch(self, feature: np.ndarray, Y: np.ndarray): # OK

        sorted_feature = np.sort(feature)
        means = np.zeros(len(feature) - 1)
        for i in range(len(feature) - 1):
            means[i] = (sorted_feature[i] + sorted_feature[i + 1]) / 2
        means = np.unique(means)

        gini_impurities = np.zeros(len(means))

        for i, mean in enumerate(means):

            options = np.zeros((self.n_labels, 2)) # true / false for (n_labels)

            for x, label in zip(feature, Y):

                if (x < mean):
                    options[int(label), 0] += 1
                else:
                    options[int(label), 1] += 1
            weights = [np.sum(options[:, 0]) / np.sum(options), np.sum(options[:, 1]) / np.sum(options)]

            impurity = np.zeros(2)
            for (j) in range(2):
                s = np.sum(options[:, j])
                if s > 0:
                    options[:, j] /= s
                    impurity[j] = 1 - np.sum(options[:, j]**2)
            impurity = (impurity[0] * weights[0]) + (impurity[1] * weights[1])
            gini_impurities[i] = impurity

        min_impurity, branch_condition = np.min(gini_impurities), means[np.argmin(gini_impurities)]

        return min_impurity, branch_condition
    
    def split_data(self, X, Y, feature, threshold):

        mask = X[:, feature] < threshold 

        X_left = X[mask] # true
        Y_left = Y[mask]

        X_right = X[~mask] # false
        Y_right = Y[~mask]

        return X_left, Y_left, X_right, Y_right
    
    def predict(self, X, Y, node: Node, predicted):

        if (node.value != None):
            return np.sum(Y == node.value)
        
        feature = node.feature
        threshold = node.threshold
        X_left, Y_left, X_right, Y_right = self.split_data(X, Y, feature=feature, threshold=threshold)
        predicted += self.predict(X_left, Y_left, node.left, predicted)
        predicted += self.predict(X_right, Y_right, node.right, predicted)


model = ClassificationTree()
model.n_labels = len(np.unique(Y_train))
model.root = model.build(X_train, Y_train)
predicted = 0
predicted = model.predict(X_test, Y_test, model.root, predicted)

print(predicted / len(Y_test))




