from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other: Value):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other: Value):

        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):

        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad = (1 - (t**2)) * out.grad
        
        out._backward = _backward
            
        return out
    

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a+b; e.label = 'e'
d = e=c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

def lol():

    h = 0.000001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'

    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e+c; d.label = 'd'
    f = Value(-2.0 + h, label='f')
    L = d * f; L.label = 'L'

    L2 = L.data

    dL_da = (L2 - L1)/h
    print(dL_da)

def lol1():

    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label='w2')

    b = Value(6.8813735870195432, label='b')

    x1w1 = x1*w1; x1w1.label='x1*w1'
    x2w2 = x2*w2; x2w2.label='x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 +  b; n.label='n'
    o = n.tanh(); o.label='o'

    o.grad = 1.0
    o._backward()
    n._backward()
    print(b.grad)

lol1()

