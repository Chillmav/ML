import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:

    def __init__(self, max_degree):

        self.max_degree = max_degree
        self.weights = np.array([])
        self.bias = 0

    def fit(self):

        x, y = self.generate_data()
        bic_values = {}
        plt.scatter(x, y, color="blue")

        for deg in range(1, self.max_degree + 1):

            power = deg
            X = x**power
            power -= 1
            while power > 0:
                 X = np.c_[X, x**power]
                 power -= 1
            X = np.c_[X, np.ones(len(x))]

            theta = np.linalg.inv(X.T @ X) @ X.T @ y
            pred = X @ theta
            residuals = y - pred
            RSS = residuals.T @ residuals

            bic_values[f"{deg}"] = {"bic": self.calc_bic(len(x), RSS, deg+2), "theta": theta, "X": X}
            plt.plot(x, pred, color="gray")
        
        sorted_bic = dict(sorted(bic_values.items(), key=lambda item: item[1]["bic"]))
        best_key = next(iter(sorted_bic))
        best_pred = sorted_bic.get(best_key).get("X") @ sorted_bic.get(best_key).get("theta")
        plt.plot(x, best_pred, color="green")
        plt.show()
    
    def generate_data(self, data_size=100):

        x = np.linspace(-1, 1, data_size)
        y = (-2 * (x ** 3)) + (3 * (x ** 2)) + 1 
        y += np.random.normal(0, 1, data_size)
        return x, y
    
    def calc_bic(self, n, RSS, k):
        return n * np.log(RSS / n) + k * np.log(n)
    
model = PolynomialRegression(max_degree=20)

model.fit()



        