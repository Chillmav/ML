import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Line:

    def __init__(self):

        self.a_LS = 0 #slope
        self.b_LS = 10 #intercept
        self.a_grad = 1
        self.b_grad = 0
        self.SSR = 0

    def calculate_SSR(self, data):

        x, y = data

        for x, y in zip(x, y):

            self.SSR += (y - (self.a_LS * x + self.b_LS))**2

    def visualize_fit(self, data):

        x, y = data
        plt.scatter(x, y)
        plt.plot(x, self.a_LS * x + self.b_LS, color="red")
        plt.plot(x, self.a_grad * x + self.b_grad, color="green")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def least_squares(self, data):

        x, y = data
        x = np.c_[x, np.ones(len(x))]
        theta = np.linalg.inv(x.T @ x) @ x.T @ y
        self.a_LS = theta[0]
        self.b_LS = theta[1]

    def gradient_descent(self, data):

        previous_mean_error = 0
        x, y = data
        learning_rate = 0.01

        for _ in range(10000):
            a_grad = 0
            b_grad = 0
            mean_error = 0
            for x_i, y_i in zip(x, y):
                mean_error += (y_i - (self.a_grad * x_i + self.b_grad))**2
                a_grad += -2 * (y_i - (self.a_grad * x_i + self.b_grad)) * x_i
                b_grad += -2 * (y_i - (self.a_grad * x_i + self.b_grad))

            mean_error /= len(x)
            a_grad /= len(x)
            b_grad /= len(x)
            self.a_grad -= a_grad * learning_rate
            self.b_grad -= b_grad * learning_rate
            if (abs(mean_error - previous_mean_error) < 1e-6):
                break
            previous_mean_error = mean_error

def generate_data(n):
    x = np.arange(n) + np.random.normal(0, 3, n)
    y = 2 * np.arange(n) + np.random.normal(0, 3, n)
    return x, y

data = generate_data(15)

line = Line()
line.calculate_SSR(data)
line.least_squares(data)
line.gradient_descent(data)
line.visualize_fit(data)

    