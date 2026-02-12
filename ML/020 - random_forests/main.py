import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ML/data/Iris.csv")
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
Y = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).values

data = np.c_[X, Y]
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
        self.value = value


class RandomForest:

    def __init__(self, trees=100, labels=3):

        self.columns = 0
        self.trees = trees
        self.labels = labels
        self.forest = []

    def build_forest(self, X, Y):

        self.columns = int(X.shape[1])
        results = [] # forest, accuracy with oob data, columns used to build trees, matrix of size (self.columns, 3)

        for v in range(1, self.columns + 1):
            forest = [] # self.trees roots

            votes = {i: [] for i in range(len(Y))}
            for _ in range(self.trees):

                variables = sorted(np.random.choice(np.arange(self.columns), v, replace=False))
                x = X[:, variables]
                bootstrapped_X, bootstrapped_Y, oob_X, oob_Y, indices = self.bootstrap(x, Y)
                tree = self.build_tree(bootstrapped_X, bootstrapped_Y)
                self.calc_predicted(tree, oob_X, oob_Y, votes, indices)
                forest.append(tree)    

            accuracy = self.eval_votes(votes, Y)
            results.append([forest, accuracy, v])
        accuracies = [r[1] for r in results]
        idx = np.argmax(accuracies)
        result = results[idx]
        self.forest = result[0]
        return result[0], result[1], result[2]
    
    def eval_votes(self, votes: dict, Y):
        
        n = len(Y)
        predictions = np.zeros(n)

        for key, value in votes.items():
            if len(value) == 0:
                predictions[key] = -1
                continue

            values, counts = np.unique(value, return_counts=True)
            most_common = values[np.argmax(counts)]
            predictions[key] = most_common

        mask = predictions != -1
        return np.sum(predictions[mask] == Y[mask]) / np.sum(mask)
    
    def build_tree(self, X: np.ndarray, Y: np.ndarray):
        
        if (len(np.unique(Y)) == 1):
            return Node(value=int(Y[0]))

        if len(Y) == 0:
            return None

        feature, threshold, _ = self.build_node(X, Y)
        node = Node(feature, threshold)
        X_right, Y_right, X_left, Y_left = self.split_data(node, X, Y)

        if len(Y_left) == 0 or len(Y_right) == 0:
            majority = int(np.bincount(Y.astype(int)).argmax())
            return Node(value=majority)
        
        node.right = self.build_tree(X_right, Y_right)
        node.left = self.build_tree(X_left, Y_left)

        return node
    
    def build_node(self, X, Y):
        
        results = np.zeros((X.shape[1], 3)) # val: [feature, threshold, impurity]

        for feature in range(X.shape[1]):

            sorted_feature = np.sort(X[:, feature].flatten())
            thresholds = np.zeros(len(sorted_feature) - 1)
            
            for i in range(len(sorted_feature) - 1):
                thresholds[i] = (sorted_feature[i] + sorted_feature[i+1]) / 2
            thresholds = np.unique(thresholds)
            
            impurities = np.zeros(len(thresholds))

            for j, threshold in enumerate(thresholds):

                options = np.zeros((self.labels, 2)) # 3 labels 2 bools
                samples = len(Y)

                for x, y in zip(X[:, feature], Y):

                    if x > threshold:
                        options[int(y), 0] += 1 # True
                    else:
                        options[int(y), 1] += 1 # False
                
                true_samples = np.sum(options[:, 0])
                false_samples = np.sum(options[:, 1])
                weight_true = true_samples / samples
                weight_false = false_samples / samples

                impurity_true = 1
                impurity_false = 1

                for label in range(self.labels):
                    if true_samples > 0:
                        impurity_true -= (options[label, 0] / true_samples)**2
                    if false_samples > 0:
                        impurity_false -= (options[label, 1] / false_samples)**2

                impurities[j] = impurity_true*weight_true + impurity_false*weight_false

            idx_of_min_impurity = np.argmin(impurities)
            threshold = thresholds[idx_of_min_impurity]
            impurity = impurities[idx_of_min_impurity]

            results[feature, 0] = feature
            results[feature, 1] = threshold
            results[feature, 2] = impurity

        idx = np.argmin(results[:, 2])
        return results[idx]

    def bootstrap(self, X: np.ndarray, Y: np.ndarray):

        bootstrap_size = X.shape[0]
        indices = np.random.randint(0, bootstrap_size, bootstrap_size)
        bootstrapped_X = X[indices]
        bootstrapped_Y = Y[indices]

        oob_mask = np.ones(bootstrap_size, dtype=bool)
        oob_mask[indices] = False

        oob_X = X[oob_mask]
        oob_Y = Y[oob_mask]
        indices = np.flatnonzero(oob_mask)

        return bootstrapped_X, bootstrapped_Y, oob_X, oob_Y, indices
    
    def calc_predicted(self, node: Node, oob_X, oob_Y, votes: dict, indices):

        for i, (x, y) in enumerate(zip(oob_X, oob_Y)):
            index = indices[i]
            votes[index].append(self.sample_prediction(node, x, y))
        
    def sample_prediction(self, node, x, y):

        if (node.value != None):
            return int(node.value)
        if (x[int(node.feature)] > node.threshold):
            return self.sample_prediction(node.right, x, y)
        else:
            return self.sample_prediction(node.left, x, y)

    def split_data(self, node: Node, X, Y):

        feature = int(node.feature)
        threshold = node.threshold
        
        mask = X[:, feature] > threshold

        X_right = X[mask]
        Y_right = Y[mask]
        X_left = X[~mask]
        Y_left = Y[~mask]

        return X_right, Y_right, X_left, Y_left

    def predict(self, X, Y):
        
        n = len(Y)
        predicted = 0
        for x, y in zip(X, Y):
            real = int(y)
            choices = np.zeros(self.labels)
            for tree in self.forest:
                choices[self.sample_prediction(tree, x, y)] += 1
            
            label = np.argmax(choices)
            
            if (int(label) == int(real)):
                predicted += 1
        return predicted / n
    
model = RandomForest(labels=len(np.unique(Y_train)))
forest, accuracy_oob, features = model.build_forest(X_train, Y_train)
accuracy = model.predict(X_test, Y_test)
print(accuracy_oob)
print(features)