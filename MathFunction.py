import numpy as np
from typing import Callable

def deriv(func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta: float = 0.001) -> np.ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)
def MatMul_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # Checking dimensions for matrix multiplication
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    return N

def Matrix_backward(X: np.ndarray, W: np.ndarray, sigma) -> np.ndarray:
    assert X.shape[1] == W.shape[0]
    N = np.dot(X, W)
    S = sigma(N)

    dSdN = deriv(sigma, N)
    dNdX = deriv(W, (1,0))

    return np.dot(dSdN, dNdX)