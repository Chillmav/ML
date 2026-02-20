import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X0 = np.random.normal(0, 1, 400).reshape(-1, 1)
X1 = np.random.normal(3, 1, 600).reshape(-1, 1)
X2 = np.random.normal(5, 1, 400).reshape(-1, 1)
X = np.concatenate([X0, X1, X2])
Y_0 = np.zeros(400, dtype=int)
Y_1 = np.ones(600, dtype=int)
Y_2= np.full(400, 2, dtype=int)
Y = np.concatenate([Y_0, Y_1, Y_2])

data = np.c_[X, Y]

def split_dataset(data: np.ndarray):

    size = data.shape[0]
    train_data_indices = np.random.choice(size, int(np.round(size * 4/5)), replace=False)
    test_data = np.delete(data, train_data_indices, axis=0)
    train_data = data[train_data_indices]
    return train_data, test_data

train_data, test_data = split_dataset(data)
X_train, Y_train, X_test, Y_test = train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]


class SoftmaxRegression:

    def __init__(self, epochs = 1000, lr=0.01):

        self.epochs = 1000
        self.theta = []
        self.lr = lr
        self.labels = 0

    def train(self, X, Y):

        self.labels = len(np.unique(Y))
        X = np.c_[X, np.ones(X.shape[0])]
        self.theta = np.zeros((self.labels, X.shape[1]))
        Y = self.one_hot(Y)

        for l in range(self.labels):
            self.theta[l] = np.random.uniform(-1, 1, size=X.shape[1])

        for e in range(self.epochs):
            gradient = (self.softmax(X) - Y).T @ X
            self.theta -= self.lr * (gradient / Y.shape[0])


        print(self.theta)
    
    def test(self, X, Y):

        X = np.c_[X, np.ones(X.shape[0])]
        predicted = self.softmax(X)
        predicted = np.argmax(predicted, axis=1)
        accuracy = np.sum(Y == predicted) / len(Y)
        return accuracy
    
    def softmax(self, X):

        P = np.zeros((X.shape[0], self.labels))

        for i in range(X.shape[0]):

            z = np.zeros(self.labels)

            # logits
            for l in range(self.labels):
                z[l] = X[i] @ self.theta[l]

            # stabilność
            z = z - np.max(z)

            exp_z = np.exp(z)

            P[i] = exp_z / np.sum(exp_z)

        return P
    
    def one_hot(self, Y):

        Y_one_hot = np.zeros((len(Y), self.labels), dtype=int)

        for i, y in enumerate(Y):
            Y_one_hot[i, int(y)] = 1

        return Y_one_hot
    
model = SoftmaxRegression()

model.train(X_train, Y_train)
acc = model.test(X_test, Y_test)
print(acc)