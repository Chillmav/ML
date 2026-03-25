import numpy as np
import math
from value import Value
from tensor import Tensor
from data import X, y
import matplotlib.pyplot as plt

# Next steps:
# 1 Visualization of training 
# 2 Visualization the decision line
# 3 Cross Entropy(Binary) (sigmoid output)
# 4 Cross Entropy(Multiple classes) (softmax)
# 5 Adam Optimizer

class Layer:

    def __init__(self, inputs: int, outputs: int, is_lin=False, activation_function="ReLU"): # outputs -> number of neurons in layer

        self.W = Tensor(data=np.random.uniform(-1, 1, (inputs, outputs)))
        self.b = Tensor(data=np.random.uniform(-1, 1, (1, outputs)))
        self.is_lin = is_lin
        self.activation_function = activation_function

    def __call__(self, X):

        if not isinstance(X, Tensor):
            X = Tensor(X)

        Y = X @ self.W + self.b

        if self.activation_function == "ReLU":
            return Y.relu() if not self.is_lin else Y
        elif self.activation_function == "tanh":
            return Y.tanh() if not self.is_lin else Y
        elif self.activation_function == "sigmoid":
            return Y.sigmoid() if not self.is_lin else Y
        
    def parameters(self):

        return [self.W, self.b]
    
class MLP:

    def __init__(self, inputs: int, n_outputs: np.ndarray): # n_outputs -> list of numbers of neurons in layers
        
        sz = [inputs] + n_outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]
    
    def __call__(self, X: np.ndarray):

        for layer in self.layers:
            X = layer(X)
        return X

    def activation_output(self, activation=None):

        if activation is None:
            self.linear_output()
        elif activation == "sigmoid":
            self.sigmoid_output()

    def linear_output(self):

        self.layers[-1].is_lin = True

    def sigmoid_output(self):

        self.layers[-1].activation_function = "sigmoid"

    def parameters(self):

        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):

        for layer in self.layers:

            layer.W.grad = np.zeros_like(layer.W.grad)
            layer.b.grad = np.zeros_like(layer.b.grad)
    
    def MSE(self, y_real, y_hat: Tensor):

        return ((y_hat - y_real)**2).sum() / len(y_real.data)
    
    def BCE(self, y_real, y_hat: Tensor): # Binary Cross Entropy

        eps = 1e-8
        y_hat = y_hat * (1 - 2*eps) + eps

        return -(y_real * y_hat.log() + (1 - y_real) * (1 - y_hat).log()).sum() / y_real.data.shape[0]
        

    def train(self, X, y, epoch=1000, lr=0.01, spe=2):

        X_full = X
        y_full = y
        losses = np.zeros(epoch)

        for e in range(epoch):

            for s in range(spe):

                X_batch, y_batch = self.SGD(X_full, y_full)

                y_hat = self(X_batch)
                
                loss = self.BCE(y_real=y_batch, y_hat=y_hat)
                
                losses[e] += loss.item() / spe
                self.zero_grad()
                loss.backward()

                for param in self.parameters():
                    param.data -= param.grad * lr

            if e % 20 == 0:
                print(f"{e} : {loss.item():.6f}")

        self.visualize(epoch, losses)

    def visualize(self, epoch, losses):

        plt.plot(np.arange(0, epoch), losses)
        plt.xlabel("Epochs")
        plt.ylabel("BCE")
        plt.show()

    def SGD(self, X, y, size=10):

        X_data = X.data if isinstance(X, Tensor) else X
        y_data = y.data if isinstance(y, Tensor) else y

        indices = np.random.choice(len(X_data), size, replace=False)

        X_batch = X_data[indices]
        y_batch = y_data[indices].reshape(-1, 1)

        return Tensor(X_batch), Tensor(y_batch)
    

nn = MLP(2, [3, 3, 1])
nn.sigmoid_output()
nn.train(X, y)

