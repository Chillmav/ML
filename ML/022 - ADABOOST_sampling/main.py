import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ML\data\obesity_data.csv")

X = df[["Age","Height","Weight","BMI","PhysicalActivityLevel"]].values

Y = df["ObesityCategory"].map({
    "Underweight": 0,
    "Normal weight": 1,
    "Overweight": 2,
    "Obese": 3
}).values

age = df["Gender"].map({
    "Male": 0,
    "Female": 1
}).values

data = np.c_[X, age]
data = np.c_[data, Y]

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


class ADABOOST:

    def __init__(self, stumps = 100):

        self.stumps = stumps
        self.labels = 0
        self.forest = []

    def build_forest(self, X: np.ndarray, Y: np.ndarray):

        self.labels = len(np.unique(Y))
        sample_weights = np.full(len(Y), fill_value=(1/len(Y)))
        stump, sample_weights = self.build_stump(X, Y, sample_weights)
        # for i in range(self.stumps):

        #     stump, sample_weights = self.build_stump(X, Y, sample_weights)
        #     self.forest.append(stump)


        
    def build_stump(self, X: np.ndarray, Y: np.ndarray, sample_weights):

        results = np.zeros((X.shape[1], 3)) # row -> [feature, impurity, threshold]

        for feature in range(X.shape[1]):
            
            X_sorted = np.sort(X[:, feature])
            thresholds = (X_sorted[:-1] + X_sorted[1:]) / 2
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

        print(results)
        
        return Node(), []

model = ADABOOST(100)

model.build_forest(X_train, Y_train)

    



