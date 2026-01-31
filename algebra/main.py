import numpy as np
import scipy.linalg as la


A = np.array([
    [1, 1],
    [1, 1]
])

B = np.array([
    [2],
    [2]
], dtype=float)

y = la.solve(A, B)
print(y)