import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split(X, Y):

        data = np.c_[X, Y]
        size = data.shape[0]
        train_data_indices = np.random.choice(size, int(np.round(size * 3/5)), replace=False)
        test_data = np.delete(data, train_data_indices, axis=0)
        train_data = data[train_data_indices]

        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

def data_1D(n=100):

    x = np.linspace(0, 10, n)
    std_2 = np.std(x) / 2
    mean = np.mean(x)
    labels = [-1 if (xi < mean - std_2 or xi > mean + std_2) else 1 for xi in x]

    return x, labels


X, Y = data_1D(100)
X_train, Y_train, X_test, Y_test = split(X, Y)

class SVM:
     
     def __init__(self, kernel="linear", option="hard margin sum"):
          
          self.kernel = kernel
          self.option = option
          # Hard margin sum assums that labels can be separeted perfectly


     def train(self, X, Y):
          pass