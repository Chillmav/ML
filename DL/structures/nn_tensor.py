import numpy as np
import math
from value import Value

class Neuron:

    def __init__(self, inputs: int, is_lin=False):

        self.w = [Value(np.random.uniform(-0.1, 0.1)) for _ in range(inputs)]
        self.b = Value(np.random.uniform(-0.1, 0.1))
        self.is_lin = is_lin

    def __call__(self, x: np.ndarray):

        y = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return y.sigmoid() if not self.is_lin else y

    def parameters(self):

        return self.w + [self.b]

class Layer:

    def __init__(self, inputs: int, outputs: int): # outputs -> number of neurons in layer

        self.neurons = [Neuron(inputs) for _ in range(outputs)]
    
    def __call__(self, x: np.ndarray):

        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):

        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:

    def __init__(self, inputs: int, n_outputs: np.ndarray): # n_outputs -> list of numbers of neurons in layers

        sz = [inputs] + n_outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]
    
    def __call__(self, x: np.ndarray):

        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):

        return [p for layer in self.layers for p in layer.parameters()]

    def linear_output(self):

        for neuron in self.layers[-1].neurons: neuron.is_lin = True

            

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, 0, 0, 1.0] #desired outputs

nn = MLP(3, [4, 4, 1])
nn.linear_output()
lr = 0.1

for e in range(300):

    y = [nn(x) for x in xs]
    loss = sum([(y_r - y_p)**2 for y_r, y_p in zip(ys, y)])
    if (e % 20 == 0):
        print(f"{e} : {loss.data}")

    for p in nn.parameters():
        p.grad = 0.0

    loss.backward()
    for param in nn.parameters():
        param.data -= param.grad * lr
        
print(ys)
print([nn(x).data for x in xs])
