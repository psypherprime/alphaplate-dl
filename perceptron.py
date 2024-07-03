import numpy as np

def forward_linear_regression(X_batch: np.ndarray, y_batch: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[float, dict[str, np.ndarray]]:
    # Checking if the number of labels is equal to number of inputs
    assert X_batch.shape[0] == y_batch.shape[0]
    # Checking the dimensions for matrix multiplication
    assert X_batch.shape[1] == y_batch.shape[0]
    # Checking that B is 1x1 ndarray
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    N = np.dot(X_batch, weights['W'])
    P = N + weights['B']

    loss = np.mean(np.square(y_batch - P))
    forward_info: dict[str, np.ndarray] = {'X': X_batch, 'N': N, 'P': P, 'y': y_batch}

    return loss, forward_info