import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


diabetes = load_diabetes()
X = diabetes.data
Y = diabetes.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

class XGBoost:

    def __init__(self, lr=.3, trees=100, gamma=0, mag_depth=6):

        self.scaler = StandardScaler()
        pass
    

    def fit(self, X, Y):

        X = self.scaler.fit_transform(X)

    def predict(self, X):

        predictions = np.zeros(X.shape[0])

        return predictions

    def SSR(self, predictions, real):

        return np.sum((predictions - real)**2)

    def evaluate(self, predictions, real):

        SSR = self.SSR(predictions, real)
        pass

    

