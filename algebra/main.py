import numpy as np
import scipy.linalg as la

Q = np.array([[1, -2], [1, 5]])
L = np.diag([4**10, (-3)**10])
Q_inv = np.array([[5/7, 2/7], [-1/7, 1/7]])
print(Q@L@Q_inv)
