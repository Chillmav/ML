import numpy as np


class LassoRegression:

    def __init__(self, features=20, alpha=2):
        
        self.weights = np.array([np.random.uniform(-1, 1) for _ in range(features + 1)]) # bias at the end
        self.dataset = self.generate_data()
        self.alpha = alpha
        print(self.dataset)
        print(self.weights)

    def fit_to_data(self, iterations=1000):
        
        X = self.dataset[:-1].T
        output = self.dataset[-1]
        for _ in range(iterations):
            for i in range(len(self.weights) - 1):
                pred = X @ self.weights[:-1] + self.weights[-1]
                feature = X[:, i]
                ro = np.sum(feature * (output - pred + (self.weights[i] * feature)))
                z = np.sum(feature ** 2)
                if (ro < -self.alpha):
                    self.weights[i] = (ro + self.alpha) / z
                elif (ro > self.alpha):
                    self.weights[i] = (ro - self.alpha) / z
                else:
                    self.weights[i] = 0 

            pred = X @ self.weights[:-1] + self.weights[-1]
            bias = np.mean(output - pred)
            self.weights[-1] = bias

        print(self.weights)

    def generate_data(self, data_length=20):
        
        dataset = []
        for _ in range(len(self.weights) - 1):
            data = np.linspace(np.random.uniform(-2, 2), np.random.uniform(5, 10), data_length) + np.full(data_length, np.random.normal(0, 1))
            dataset.append(data)
        outputs = np.zeros(data_length)
        for i in range(len(self.weights) - 1):
            outputs += dataset[i] * np.full(data_length, np.random.uniform(-2, 2))
        random_bias = np.random.uniform(-2, 2)
        outputs += random_bias
        outputs += np.random.normal(0, 1, data_length)
        dataset.append(outputs)
        return np.array(dataset)
    
lasso = LassoRegression()
lasso.fit_to_data()


