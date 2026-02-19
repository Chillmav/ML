import numpy as np
import matplotlib.pyplot as plt

X0 = np.random.normal(0, 1, 400)
X1 = np.random.normal(3, 1, 600)
X = np.concatenate([X0, X1])
Y_0 = np.zeros(400)
Y_1 = np.ones(600)
Y = np.concatenate([Y_0, Y_1])
data = np.c_[X, Y]
def split_dataset(data: np.ndarray):

    size = data.shape[0]
    train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
    test_data = np.delete(data, train_data_indices, axis=0)
    train_data = data[train_data_indices]
    return train_data, test_data

train_data, test_data = split_dataset(data)
X_train, Y_train, X_test, Y_test = train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]



class LogisticRegression:

    def __init__(self, epoch=1000, lr=0.01):
        
        self.theta = np.array([np.random.normal(-1, 1), np.random.normal(-1, 1)])
        self.epochs = 1000
        self.lr = lr

    def train(self, X, Y):
        
        for e in range(self.epochs):

            error = Y - self.sigmoid(X)

            self.theta[0] += self.lr * np.mean(error)
            for w in range(len(self.theta[1:])):
                self.theta[w+1] += self.lr * np.mean(X[:, w] * error)
        
    def test(self, X, Y):

        n = len(Y)
        predicted = np.round(self.sigmoid(X)).flatten()
        x_n = np.linspace(np.min(X), np.max(X), n)
        plt.scatter(X, Y)
        plt.plot(x_n, self.sigmoid(x_n))
        plt.show()


        return np.sum(predicted == Y) / n

    def sigmoid(self, X):

        p_X = 1 / (1 + np.exp(-(self.theta[0] + self.theta[1]*X)))

        return p_X
        
    
model = LogisticRegression()

model.train(X_train, Y_train)
print(model.test(X_test, Y_test))