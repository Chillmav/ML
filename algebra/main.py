import numpy as np
import scipy.linalg as la


A = np.array([
    [1, -0.05, -0.3],
    [-0.01, 1, -0.01],
    [-0.1, 0, 1]
])

B = np.array([
    [25],
    [10],
    [14]
], dtype=float)

y = la.solve(A, B)
print(y)