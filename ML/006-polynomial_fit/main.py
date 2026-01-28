import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:

    def __init__(self, polynomial_degree=5):

        self.pol_deg = polynomial_degree
        self.coef = np.full(self.pol_deg, np.random.uniform(-2, 2)) #  w_1 * x^n + w_2 * x^n-1 + ... + w_n * x^1
        self.bias = np.random.uniform(-2, 2)
        self.data = self.generate_data()
        pass

    def generate_data(self, data_size=60): 

        x = np.linspace(-1, 1, data_size)
        y = np.full(data_size, self.bias)

        power = self.pol_deg
        for _ in range(len(self.coef)):
            
            y += x**power * np.random.uniform(-2, 2)
            power -= 1

        print(x, y)
        y += np.random.normal(0, 1, data_size)
        return x, y

    def fit(self, regularization="None", alpha=1):

        x, y = self.data
        if (regularization == "None"):
            X = x**self.pol_deg
            power = self.pol_deg - 1
            while power > 0:
                X = np.c_[X, x**power]
                power -= 1
            X = np.c_[X, np.ones(len(x))]
            theta = np.linalg.inv(X.T @ X) @ X.T @ y
            self.coef = theta[:-1]
            self.bias = theta[-1]

        if (regularization == "Ridge"):

            X = np.vander(x, self.pol_deg + 1)
            I = np.identity(self.pol_deg + 1)
            I[-1, -1] = 0
            theta = np.linalg.inv(X.T @ X + alpha * np.identity(self.pol_deg + 1)) @ X.T @ y
            self.coef = theta[:-1]
            self.bias = theta[-1]


    def visualize_data(self):

        x, y = self.data
        pred = self.calculate_pred(x)
        plt.scatter(x, y, color="blue")
        plt.plot(x, pred, color="green")
        plt.show()
        
    def calculate_pred(self, x):

        power = self.pol_deg
        pred = np.zeros(len(x))
        coef_i = 0

        while power > 0:
            pred += np.full(len(x), self.coef[coef_i] * x**power)
            coef_i += 1
            power -= 1

        pred += np.full(len(x), self.bias)
        return pred


model = PolynomialRegression()
model.fit(regularization="Ridge", alpha=2)
model.visualize_data()



