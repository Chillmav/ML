import numpy as np


# 1

P = np.array([0.7, 0.2, 0.1])
Q1 = np.array([0.6, 0.3, 0.1])
Q2 = np.array([0.2, 0.7, 0.1])

def entropy(P):

    return np.sum(P * -np.log(P))

def cross_entropy(P, Q):

    return np.sum(P * -np.log(Q))

def KL_divergence(P, Q):

    return -entropy(P) + cross_entropy(P, Q)


# print(entropy(P))
# print(cross_entropy(P, Q1))
# print(cross_entropy(P, Q2))
# print(KL_divergence(P, Q1))
# print(KL_divergence(P, Q2))
# print(entropy(P) + KL_divergence(P, Q1))
# print(cross_entropy(P, Q1))

# From outputs I can deduce that Q1 is more similar to P than Q2 is similar to P.

# 2

# samples = np.random.choice(len(P), size=10000, p=P)
# print(np.sum(samples == 0.6))

# def sampling_method(Q, samples):

#     return np.mean(-np.log(Q[samples]))

# print(sampling_method(Q1, samples))
# print(cross_entropy(P, Q1))

# 3

np.random.seed(42)

P = np.array([0.5, 0.3, 0.2])
data = np.random.choice(len(P), size=1000, p=P)

def CE_minimizing(data):

    values, counts = np.unique(data, return_counts=True)
    theta = np.zeros(len(P))

    for k in range(len(theta)):

        theta[k] = counts[k] / np.sum(counts)

    print(theta)
    print(cross_entropy(P, theta))
    print(cross_entropy(P, np.array([0.4, 0.4, 0.2])))



CE_minimizing(data)