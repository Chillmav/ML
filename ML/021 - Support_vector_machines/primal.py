import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42)

n = 100

x1_class1 = np.random.uniform(600, 900, n)
x1_class2 = np.random.uniform(100, 400, n)

x2_class1 = np.random.uniform(6, 9, n)
x2_class2 = np.random.uniform(1, 4, n)

X_class1 = np.column_stack((x1_class1, x2_class1))
X_class2 = np.column_stack((x1_class2, x2_class2))

x1_class1_t = np.random.uniform(600, 900, n // 5)
x1_class2_t = np.random.uniform(100, 400, n // 5)

x2_class1_t = np.random.uniform(6, 9, n // 5)
x2_class2_t = np.random.uniform(1, 4, n // 5)

X_class1_t = np.column_stack((x1_class1_t, x2_class1_t))
X_class2_t = np.column_stack((x1_class2_t, x2_class2_t))

X = np.vstack((X_class1, X_class2))
Y = np.hstack((np.ones(n), np.zeros(n)))
X_test = np.vstack((X_class1_t, X_class2_t))
Y_test = np.hstack((np.ones(n//5), np.zeros(n//5)))

# plt.scatter(X_class1[:,0], X_class1[:,1])
# plt.scatter(X_class2[:,0], X_class2[:,1])



class SVM:
    
     def __init__(self, regularization_param=10, lr=0.01, epochs=1000):
     
          self.params = 0
          self.epochs = epochs
          self.lr = lr
          self.C = regularization_param
          self.mean = 0
          self.std = 0

     def train(self, X: np.ndarray, Y: np.ndarray):

          X = self.standardization(X)
          Y = self.label_Y(Y)
          self.params = np.random.uniform(0, 1, X.shape[1] + 1) # bias as last param

          for _ in range(self.epochs):
               
               total_grad_w = self.params[:-1].copy()
               total_grad_b = 0
               for i in range(len(Y)):
                    
                    if (Y[i]*(np.dot(self.params[:-1], X[i, :]) + self.params[-1])) <= 1:
                         total_grad_w -= self.C * (Y[i]*X[i, :])
                         total_grad_b += self.C * (-Y[i])
               
               total_grad_w /= len(Y)
               total_grad_b /= len(Y)
               self.params[:-1] -=  self.lr * total_grad_w
               self.params[-1] -= self.lr * total_grad_b
          
          self.plot_graph(X)

     def test(self, X_test, Y_test):

          X_test = self.standardization(X_test, train=False)
          Y_test = self.label_Y(Y_test)

          predictions = np.sign(X_test @ self.params[:-1] + self.params[-1])
          predictions = self.label_Y(predictions)

          print(np.sum(predictions == Y_test) / len(Y_test))
          return np.sum(predictions == Y_test) / len(Y_test)
     
     def standardization(self, X, train=True):

          if train:
               self.mean = np.mean(X, axis=0)
               self.std = np.std(X, axis=0)

          return (X - self.mean) / self.std
     
     def plot_graph(self, X):

          X_N = []
          for i in range(len(self.params) - 1):
               x = np.arange(np.min(X[:, i]), np.max(X[:, i]), (np.max(X[:, i]) - np.min(X[:, i])) / 100)
               if len(X_N) == 0:
                    X_N = x
               else:
                    X_N = np.vstack((X_N, x))
          X_N = X_N.T
          plt.scatter(X[:, 0], X[:, 1])

          x1 = X_N[:, 0]
          x2 = -(self.params[0]*X_N[:, 0] + self.params[-1]) / self.params[1]

          plt.plot(x1, x2)
          plt.plot(x1, -(self.params[0]*X_N[:, 0] + self.params[-1] - 1) / self.params[1], linestyle="dashed", color="orange")
          plt.plot(x1, -(self.params[0]*X_N[:, 0] + self.params[-1] + 1) / self.params[1], linestyle="dashed", color="orange")
          plt.show()

     def label_Y(self, Y):

          inherent_labels = np.unique(Y)
          if len(inherent_labels) > 2:
               raise Exception("This is binary SVM so no more than 2 classes are permitted")
          Y[Y == 0] = -1
           
          return Y


model = SVM()
model.train(X, Y)
model.test(X_test, Y_test)
