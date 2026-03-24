import numpy as np
import math
from value import Value
from tensor import Tensor
from data import X, y

class Layer:

    def __init__(self, inputs: int, outputs: int, is_lin=False): # outputs -> number of neurons in layer

        self.W = Tensor(data=np.random.uniform(-0.1, 0.1, (inputs, outputs)))
        self.b = Tensor(data=np.random.uniform(-0.1, 0.1, (1, outputs)))
        self.is_lin = is_lin

    def __call__(self, X):

        if not isinstance(X, Tensor):
            X = Tensor(X)

        Y = X @ self.W + self.b
        return Y.relu() if not self.is_lin else Y
    
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

    def linear_output(self):

        self.layers[-1].is_lin = True

    def parameters(self):

        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):

        for layer in self.layers:

            layer.W.grad = np.zeros_like(layer.W.grad)
            layer.b.grad = np.zeros_like(layer.b.grad)
    
    def SSE(self, y_real, y_predicted: Tensor):

        return ((y_predicted - y_real)**2).sum()
    
    def train(self, X, y, epoch=1000, lr=0.01):

        X = Tensor(X) if not isinstance(X, Tensor) else X
        y = Tensor(y) if not isinstance(y, Tensor) else y

        for e in range(epoch):

            y_pred = self(X)
            loss = self.SSE(y_real = y, y_predicted=y_pred)
            if (e % 20 == 0):
                print(f"{e} : {loss.data}")
            
            self.zero_grad()

            loss.backward()
            for param in self.parameters():
                param.data -= param.grad * lr

nn = MLP(4, [10, 10, 1])
nn.train(X, y)

