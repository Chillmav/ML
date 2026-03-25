import numpy as np

def generate_data(n=200, noise=0.8):

    # half samples per class
    n_half = n // 2

    # Class 0 (label 0)
    X0 = np.random.randn(n_half, 2) * noise + np.array([-2, -2])
    y0 = np.zeros(n_half)

    # Class 1 (label 1)
    X1 = np.random.randn(n_half, 2) * noise + np.array([2, 2])
    y1 = np.ones(n_half)

    # Combine
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y

# Example usage
X, y = generate_data(n=200, noise=1.0)

