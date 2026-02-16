import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ML/data/obesity_data.csv")
X = df[["Age", "Height", "Weight", "BMI", "PhysicalActivityLevel"]].values
age = df["Gender"].map({
    "Male": 0,
    "Female": 1
}).values

Y = df["ObesityCategory"].map({
    "Normal weight": 0,
    "Overweight": 1,
    "Obese": 2,
    "Underweight": 3
}).values

X = np.c_[X, age]
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

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, discrete=False):

        self.feature = feature
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.value = value
        self.discrete = discrete

class AdaBoost:

    def __init__(self, stumps=100, labels=4):

        self.columns = 0
        self.stumps = stumps
        self.labels = labels
        self.forest = []

    def build_forest(self, X, Y):

        init_sample_weights = 1 / X.shape[0]
        print(init_sample_weights)

    def build_stump(self, X: np.ndarray, Y: np.ndarray, sample_weights: np.ndarray = []):

        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_discrete = False
        results = np.zeros((X.shape[1], 4)) # val: [feature, threshold, impurity, isDiscrete]

        for feature in range(X.shape[1]):

            values = X[:, feature]
            unique_vals = np.unique(values)

            if (len(unique_vals) > 2): # continous feature:

                # Sort feature and labels
                sorted_idx = np.argsort(values)
                sorted_values = values[sorted_idx]
                sorted_labels = Y[sorted_idx]
                thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2 # wow
                thresholds = np.unique(thresholds)

                impurities = np.zeros(len(thresholds))

                for j, threshold in enumerate(thresholds):

                    options = np.zeros((self.labels, 2))
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
                results[feature, 3] = 0 # for continous data


            else: # boolean feature (gender for example):
                print(len(np.unique(feature)))
                options = np.zeros((self.labels, 2))
                samples = len(Y)

                for (x, y) in zip(X[:, feature], Y):
                    
                    if x == X[0, feature]:
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
                    
                    impurity = impurity_true*weight_true + impurity_false*weight_false
                    threshold = X[0, feature]

                    results[feature, 0] = feature
                    results[feature, 1] = threshold
                    results[feature, 2] = impurity
                    results[feature, 3] = 1 # for discrete data

        idx = np.argmin(results[:, 2])   
        print(results[idx])
        return results[idx]

model = AdaBoost(100, len(np.unique(Y)))

model.build_forest(X_train, Y_train)
model.build_stump(X_train, Y_train)





