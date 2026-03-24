from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt

class Tensor:

    def __init__(self, data, _children=(), _op='', label=''):

        self.data = np.array(data) if not isinstance(data, np.ndarray) else data # (batch_size, features)

        if self.data.shape == ():
            self.data = self.data.reshape(1, 1)

        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):

        return f"Tensor(data={self.data})"
    
    def __add__(self, other: Tensor):

        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():

            self.grad += out.grad

            if other.data.shape == out.grad.shape:
                other.grad += out.grad
            else:
                other.grad += out.grad.sum(axis=0)

        out._backward = _backward

        return out
    
    def __mul__(self, other: Tensor):

        other = other if isinstance(other, Tensor) else Tensor(other)
    
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad

            grad_other = self.data * out.grad

            if other.grad.shape != grad_other.shape:
                grad_other = grad_other.sum(axis=0, keepdims=True)

            other.grad += grad_other

        return out
    
    def __rmul__(self, other):

        return self * other
    
    def __matmul__(self, other: Tensor):
        
        other = other if isinstance(other, Tensor) else Tensor(other) # other -> (n, 1) (weights)

        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out
    
    def __pow__(self, other):

        assert isinstance(other, (int, float))

        out = Tensor(self.data**other, (self, ), f"**{other}")

        def _backward():

            self.grad += other * (self.data**(other-1)) * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):

        out = Tensor(np.where(self.data < 0, 0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += np.where(self.data < 0, 0, 1) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        
        n = self.data
        e = np.exp(2*n)
        t = (e - np.ones_like(self.data)) / (e + np.ones_like(self.data))
        out = Tensor(t, (self, ), "tanh")

        def _backward():
            self.grad += (np.ones_like(self.data) - (t**2)) * out.grad
        
        out._backward = _backward
            
        return out
    
    def sigmoid(self):

        n = self.data
        t = 1 / (1 + np.exp(-n))
        out = Tensor(t, (self, ), "sigmoid")

        def _backward():
            self.grad += t * (np.ones_like(self.data) - t) * out.grad
        
        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def sum(self):
        # sum1 = np.sum(self.data, axis = 0) -> I sum up loss inside each output over samples
        # sum2 = np.sum(sum1) -> I sum up total loss over different outputs
        # total sum is just np.sum(self.data)
        out = Tensor(np.sum(self.data), (self, ), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        
        out._backward = _backward

        return out
    
    def __radd__(self, other: Tensor):

        return self + other

    def __rsub__(self, other: Tensor):
        
        return -self + other

    def __sub__(self, other: Tensor):
        
        return self + (-1 * other)

    def __neg__(self):
        return self * -1
