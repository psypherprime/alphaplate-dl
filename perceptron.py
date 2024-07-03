#%%
import numpy as np
#%%
def forward_linear_regression(X_batch: np.ndarray, y_batch: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[float, dict[str, np.ndarray]]:
    """
    Implementation of forward pass of the perceptron.
    Here, We are calculating the matrices to get the prediction label.
    """
    # Checking if the number of labels is equal to number of inputs
    assert X_batch.shape[0] == y_batch.shape[0]
    # Checking the dimensions for matrix multiplication
    assert X_batch.shape[1] == weights['W'].shape[0]
    # Checking that B is 1x1 ndarray
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    N = np.dot(X_batch, weights['W'])
    P = N + weights['B']

    loss = np.mean(np.square(y_batch - P))
    forward_info: dict[str, np.ndarray] = {'X': X_batch, 'N': N, 'P': P, 'y': y_batch}

    return loss, forward_info
#%%
def loss_gradients(forward_info: dict[str, np.ndarray], weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    batch_size  = forward_info['X'].shape[0]
    """
    dLdW = dLdP * dPdN * dNdW
    dLdP = dLdP * dPdB * dBdB
    """
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN
    dNdW = np.transpose(forward_info['X'], (1, 0)) # Check this

    dLdW = np.dot(dNdW, dLdN)
    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradients: dict[str, np.ndarray] = {'W': dLdW, 'B': dLdB}

    return loss_gradients
#%%
Batch = tuple[np.ndarray, np.ndarray]

def generate_batch(X: np.ndarray, y: np.ndarray, start: int = 0, batch_size: int = 32) -> Batch:
    # Checking for perceptron requirements and same units
    assert X.ndim == y.ndim == 2
    
    # Controlling overflow
    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start
    
    X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]
    
    return X_batch, y_batch
#%%
def forward_loss(X: np.ndarray, y:np.ndarray, weights: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    N = np.dot(X, weights['W'])
    P = N + weights['B']
    loss = np.mean(np.square(y - P))

    forward_info: dict[str, np.ndarray] = {'X': X, 'N': N, 'P': P, 'y': y}

    return forward_info, loss
#%%
def init_weights(n_in: int) -> dict[str, np.ndarray]:
    W = np.random.randn(n_in, 1)
    B = np.random.randn(1, 1)
    
    weights = {'W': W, 'B': B}
    
    return weights
#%%
def train(X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01,
          return_losses:bool = False, return_weights:bool = False, seed: int = 0
          ):

    if seed != 0:
        np.random.seed(seed)
    # Shuffling Data
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm], y[perm]

    start = 0
    weightst = init_weights(X.shape[1])

    if return_losses:
        losses = []

    for _ in range(epochs):
        if start >= X.shape[0]:
            perm = np.random.permutation(X.shape[0])
            X, y = X[perm], y[perm]
            start = 0

        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size

        forward_info, loss = forward_loss(X_batch, y_batch, weightst)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weightst)
        for key in weightst.keys():
            weightst[key] -= (learning_rate * loss_grads[key])

        # print(weightst['W'], (learning_rate * loss_grads['W']), weightst['W'] - (learning_rate * loss_grads['W']))

    if return_weights:
        return losses, weightst

    return None
#%%
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
#%%
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X = s.fit_transform(X)
#%%
divd = int(0.7*len(X))
X_train, X_test, y_train, y_test = X[:divd], X[divd:], y[:divd], y[divd:]

y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
#%%
train_info = train(X_train, y_train, epochs=1000, batch_size=32, learning_rate=0.001, return_losses=True, return_weights=True)
losses = train_info[0]
weights = train_info[1]
#%%
print(losses)
#%%
print(weights)
#%%
