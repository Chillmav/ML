import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

class Line:

    def ridge_regression(self, data, λ):

        x, y = data
        X = np.c_[x, np.ones(len(x))]
        theta = np.linalg.inv(X.T @ X + λ * np.array([[1, 0], [0, 0]])) @ X.T @ y

        visualize(theta[0], theta[1], data)


    def lasso_regression(self, data, λ, lr=0.001, n_iters=1000, stop=1e-5):

        x, y = data

        slope = 0
        intercept = np.mean(y)
        prev_slope_grad_mean = 1e6
        prev_intercept_grad_mean = 1e6

        for _ in range(n_iters):

            if (slope > 0):
                slope_mean_grad = (-2 / len(x)) * (np.sum((y - (slope * x + intercept))*x))  + λ
            else:
                slope_mean_grad = (-2 / len(x)) * (np.sum((y - (slope * x + intercept))*x))  - λ
            
            intercept_mean_grad = (-2 / len(x)) * sum(y - (slope * x + intercept))

            slope -= lr * slope_mean_grad
            intercept -= lr * intercept_mean_grad

            if (abs(prev_slope_grad_mean- slope_mean_grad) <= stop and abs(prev_intercept_grad_mean- intercept_mean_grad) <= stop):
                visualize(slope, intercept, data)
                return
            else:
                prev_slope_grad_mean = slope_mean_grad
                prev_intercept_grad_mean = intercept_mean_grad


        visualize(slope, intercept, data)
        return

    def elastic_net(self, data, λ_L2, λ_L1):

        pass

def generate_data(n):
    x = np.arange(n) + np.random.normal(0, 1, n)
    y = 2 * np.arange(n) + np.random.normal(0, 1, n)
    return x, y

def visualize(slope, intercept, data):

    x, y = data
    plt.scatter(x, y, color="blue")
    plt.plot(x, x * slope + intercept, color="green")
    plt.show()


line = Line()

line.ridge_regression(generate_data(20), 0.2)
line.lasso_regression(generate_data(20), 0.2)