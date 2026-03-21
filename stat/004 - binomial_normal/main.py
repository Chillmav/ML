import numpy as np
import matplotlib.pyplot as plt
import math


n = 1000

def draw_gaussian(n, X=[], mean=None, std=None):

    if len(X)==0:

        X = np.linspace(mean - 4*std, mean + 4*std, 1000)

    mean = np.mean(X) if mean is None else mean
    std = np.std(X) if std is None else std

    Y = np.exp(-((X - mean)**2) / (2 * std**2)) / (math.sqrt(2 * math.pi) * std)
    plt.plot(X, Y, color="green")
    plt.show()
    return Y

def draw_CDF(n, Y, X=[]):

    dx = X[1] - X[0]
    Y = np.cumsum(Y) * dx 

    if len(X) == 0:
        X = np.sort(np.random.normal(0, 1, n))

    plt.plot(X, Y)
    plt.show()

def binomial_to_gaussian(n, p, q):

    mean = n * p
    std = math.sqrt(n*p*q)
    binomial = np.random.binomial(n, p, 100020)
    values, counts = np.unique(binomial, return_counts=True)
    plt.bar(values, counts/np.sum(counts))
    X = np.sort(np.random.normal(mean, std, n))

    draw_CDF(n, draw_gaussian(n, X, mean, std), X)


binomial_to_gaussian(520, 1/52, 51/52)