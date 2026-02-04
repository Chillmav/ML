import numpy as np
import scipy.linalg as la

def inv(A: np.ndarray): # matrix 3x3

    rows = A.shape[0]
    cols = A.shape[1]
    C = np.zeros((rows, cols))

    if np.linalg.det(A) != 0:

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                C[i, j] = np.linalg.det(np.c_[np.vstack([A[0:i], A[i+1:rows]])[:, 0:j], np.vstack([A[0:i], A[i+1:rows]])[:, j+1:cols]]) * (-1)**(i+j)
    
    else:
        
        print("Inverse doesn't exist")
        return 0
    
    inv = 1/np.linalg.det(A) * C.T
    return inv
inverse = inv(np.array([[1, 4, 3],
              [2, 0, 1],
              [0, -2, -1]
              ]))
print(inverse)
print(np.linalg.inv(np.array([[1, 4, 3],
              [2, 0, 1],
              [0, -2, -1]
              ])))