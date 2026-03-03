import numpy as np
import pandas as pd


# DATA to play with
df = pd.read_csv("ML/data/Iris.csv")
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
Y = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).values

## 

# Broadcasting is the term used to describe the implicit element-by-element behaviour of operations.
# axes = dimensions in numpy !!!

# axes in np.mean:

np.mean(X, axis=0)# 0 -> collapse rows (operate vertically)
np.mean(X, axis=1) # 1 -> collapse columns (operate horizontally)


# np.where

# 1D

# x = X[:, 0]
# indices = np.where(x>5) # it returned indices
# print(x[indices])

# 2D

# x = X[0:10]
# print(x)
# indices = np.where(x > 5)
# print(x[indices]) # (array([0, 5]), array([0, 0])) so positions (0, 0) and (5, 0)

# Replacing

# x = X[:, 0]
# result = np.where(x>=5, 1, 0) # if yes supply 1 if not 0
# result2 = np.where(result == 0, -1, result) # result means value itself if condition is not fulfilled
# print(result2)

# isin

real = Y
predicted = np.random.randint(0, 3, len(Y))

mask = np.isin(predicted, [0, 1])

print(predicted[mask])

# Filtering datasets

# Removing unwanted categories

# Checking allowed values

# Comparing large arrays efficiently
