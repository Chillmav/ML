import numpy as np
import matplotlib.pyplot as plt
import time

data_size = 200
intercept = 3
slope = 2
x = np.linspace(0, 10, data_size)
y = (intercept + slope * x) + np.random.normal(-5, 5, data_size)

def split(X, Y):

        data = np.c_[X, Y]
        size = data.shape[0]
        train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
        test_data = np.delete(data, train_data_indices, axis=0)
        train_data = data[train_data_indices]

        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

X_train, Y_train, X_test, Y_test = split(x, y)

class LinearRegression:

    def __init__(self, method="LS", batch_size=5, lr=0.001, epochs=3000):

        self.params = []
        self.intercept = 0
        self.method = method
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):

        self.intercept = np.mean(y)
        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)
        shape = x.shape[1]
        if (shape < 1):
            shape = 1
        self.params = np.zeros(shape)
        if (self.method == "LS"):

            X = np.c_[X, np.ones(X.shape[0])]
            theta = np.linalg.inv(X.T @ X) @ X.T @ Y
            self.params, self.intercept = theta[:-1], theta[-1]

        elif (self.method == "BGD"):

            for _ in range(self.epochs):
                y_hat = X @ self.params.reshape(-1, 1) + self.intercept
                error = y_hat - Y
                grad_w = (X.T @ error) * (2/X.shape[0])
                grad_b = 2/X.shape[0] * np.sum(error)

                self.params -= grad_w.flatten() * self.lr
                self.intercept -= grad_b * self.lr

        elif (self.method == "SGD"):
            
            for _ in range(self.epochs):

                idx = np.random.randint(0, X.shape[0])
                x = X[idx]
                y = Y[idx]

                y_hat = x @ self.params.reshape(-1, 1) + self.intercept
                error = y_hat - y
                grad_w = (x.T @ error) * 2
                grad_b = 2 * error
                self.params -= grad_w.flatten() * self.lr
                self.intercept -= grad_b * self.lr

        elif (self.method == "Mini-batch GD"):
                        
            for _ in range(self.epochs):

                indices = np.random.randint(0, X.shape[0], self.batch_size)
                x = X[indices]
                y = Y[indices]

                y_hat = x @ self.params.reshape(-1, 1) + self.intercept
                error = y_hat - y

                grad_w = (2 / x.shape[0]) * (x.T @ error)
                grad_b = (2 / x.shape[0]) * np.sum(error)

                self.params -= grad_w.flatten() * self.lr
                self.intercept -= grad_b * self.lr
        

    def predict(self, x: np.ndarray, y: np.ndarray):
        predicted = self.params @ x.T + self.intercept
        return self.SSR(y, predicted)
    
    def SSR(self, real, predicted):
         
        return np.sum((real - predicted)**2)
    
    def plot(self, x, i, color):
        
        axes[i].plot(x, self.params[0] * x + self.intercept, color=color, linewidth=5)
        
methods = [["LS", "red"], ["BGD", "brown"], ["SGD", "green"], ["Mini-batch GD", "orange"]]

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

for i, method in enumerate(methods):

    model = LinearRegression(method[0])

    start = time.perf_counter()
    model.fit(X_train, Y_train)
    end = time.perf_counter()

    SSR = model.predict(X_test, Y_test)
    model.plot(x, i, method[1])
    axes[i].scatter(X_train, Y_train, color="blue")
    axes[i].scatter(X_test, Y_test, color="green")
    axes[i].set_title(f"{method[0]}, SSR: {SSR}, weights: {model.params}, intercept: {model.intercept}", fontsize=5)
    print(f"{method[0]}, SSR: {SSR}, weights: {model.params}, intercept: {model.intercept}, time:{end - start}")

means = {}

for _ in range(100):

    for i, method in enumerate(methods):

        model = LinearRegression(method[0])
        model.fit(X_train, Y_train)
        SSR = model.predict(X_test, Y_test)
        means[method[0]] = means.get(method[0], 0) + SSR

means = {k: v / 100 for k, v in means.items()}
print(means)
plt.show()
