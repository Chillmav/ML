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

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, aos=None):

        self.feature = feature # index
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.aos = aos

class AdaBoost:

    def __init__(self, stumps=100):

        self.columns = 0
        self.stumps = stumps
        self.labels = 0
        self.forest = []
        self.weights = []

    def build_forest(self, X, Y):

        self.weights = np.full(X.shape[0], 1 / X.shape[0])
        self.labels = len(np.unique(Y))

        for _ in range(self.stumps): # self.stumps later

            stump, total_error = self.build_stump(X, Y)

            if stump is None:
                continue

            self.forest.append(stump)

            if total_error == 0:
                break

    def build_stump(self, X: np.ndarray, Y: np.ndarray):

        n_features = X.shape[1]
        best_gini_impurities = np.zeros((n_features, 3)) # slots for label
        best_thresholds = np.zeros(n_features)

        for i in range(n_features): # n_features
            
            x = X[:, i]
            sorting_indices = np.argsort(x)
            x_sorted = x[sorting_indices]
            x_sorted_unique = np.unique(x_sorted)
            thresholds = (x_sorted_unique[:-1] + x_sorted_unique[1:])/2
            gini_impurities = np.zeros((len(thresholds), 3)) # slots for label

            for j, threshold in enumerate(thresholds):
                
                _, left_Y, left_weights, _, right_Y, right_weights = self.split_XY(x, Y, threshold)

                left_w_sum = np.sum(left_weights)
                right_w_sum = np.sum(right_weights)
                w_sum = left_w_sum + right_w_sum

                left_gini, left_label = self.calc_gini(left_Y, left_weights)
                right_gini, right_label = self.calc_gini(right_Y, right_weights)

                g_split = ((left_w_sum/w_sum) * left_gini) + ((right_w_sum/w_sum) * right_gini)
                
                gini_impurities[j] = g_split, left_label, right_label

            best_gini_idx = np.argmin(gini_impurities[:, 0])
            best_gini = gini_impurities[best_gini_idx]
            best_threshold = thresholds[best_gini_idx]
            best_gini_impurities[i] = best_gini
            best_thresholds[i] = best_threshold

        idx = np.argmin(best_gini_impurities[:, 0])
        final_gini = best_gini_impurities[idx]
        final_threshold = best_thresholds[idx]

        total_error, wrong_indices = self.calc_total_error(X, Y, idx, final_threshold, int(final_gini[1]), int(final_gini[2]))

        if total_error >= 1 - 1/self.labels:
            return None, None
        
        aos = self.calc_AoS(total_error)
        self.update_weights(np.array(wrong_indices), aos)
        stump = Node(feature=idx, threshold=final_threshold, left=final_gini[1], right=final_gini[2], aos=aos)
        
        return stump, total_error
    
    def predict(self, X_test, Y_test):

        predictions = np.zeros((len(Y_test), self.labels))

        for i in range(len(Y_test)):
            
            for stump in self.forest:

                feature = stump.feature 
                threshold = stump.threshold
                left_label = stump.left
                right_label = stump.right
                aos = stump.aos
                x = X_test[i, feature]

                if x < threshold:
                    
                    predictions[i, int(left_label)] += aos

                else:

                    predictions[i, int(right_label)] += aos
        predictions = np.argmax(predictions, axis=1)

        print(f"Accuracy: {np.sum(predictions == Y_test) / len(Y_test)}")

    def calc_total_error(self, X, Y, feature, threshold, left_label, right_label):

        total_error = 0
        x = X[:, feature]
        indices = []
        for i in range(len(Y)):

            if x[i] < threshold:

                if int(Y[i]) != int(left_label):
                    total_error += self.weights[i]
                    indices.append(i)
            else:

                if (int(Y[i]) != int(right_label)):
                    total_error += self.weights[i]
                    indices.append(i)
        
        return total_error, indices

        
    def calc_gini(self, Y, weights):

        w_per_k = np.zeros(self.labels)

        for i in range(self.labels):

            w_per_k[i] = np.sum(weights[Y == i])
        
        
        if np.sum(w_per_k) == 0:
            return 0, np.argmax(w_per_k)
        
        weighted_p_k = np.sum([(w_per_k[i] / np.sum(w_per_k))**2 for i in range(self.labels)])
        
        return 1 - weighted_p_k, np.argmax(w_per_k)

    def split_XY(self, X, Y, threshold):

        mask = X < threshold
        left_X, left_Y, left_weights = X[mask], Y[mask], self.weights[mask]
        right_X, right_Y, right_weights = X[~mask], Y[~mask], self.weights[~mask]

        return left_X, left_Y, left_weights, right_X, right_Y, right_weights
    
    def calc_AoS(self, total_error): # Calculate Amount of Say, negative amount of say flip the classification

        e = 1e-9
        return np.log((1-total_error + e)/(total_error + e)) + np.log(self.labels - 1)

    def update_weights(self, indices, aos): # indices_of_incorrectly_classified_weights

        mask = np.ones(len(self.weights), dtype=bool)
        mask[indices] = False
        rest_indices = np.arange(len(self.weights))[mask]

        for i in indices:

            self.weights[i] = self.weights[i] * np.exp(aos)
        
        for j in rest_indices:
            
            self.weights[j] = self.weights[j] * np.exp(-aos)

        sum = np.sum(self.weights)

        self.weights /= sum
    
model = AdaBoost(150)
model.build_forest(X_train, Y_train)
model.predict(X_test, Y_test)




