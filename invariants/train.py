import numpy as np
from neuralnetwork import NeuralNetwork
from optimizers import Optimizer
from copy import deepcopy


def generate_batches(X_train, y_train, batch_size) -> tuple[np.ndarray]:
    assert X_train.shape[0] == y_train.shape[0]

    N = X_train.shape[0]

    for ii in range(0, N, batch_size):
        X_batch, y_batch = X_train[ii:ii + batch_size], y_train[ii:ii + batch_size]
        yield X_batch, y_batch


def permute_data(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm], y_train[perm]
    return X_train, y_train


class Trainer(object):

    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
            epochs: int = 100, batch_size: int = 32, eval_per: int = 10, restart: bool = False,
            early_stopping: bool = False, seed: int = None):

        setattr(self.optim, "max_epochs", epochs)
        self.optim._setup_decay()
        # last_model = deepcopy(self.net)

        if seed is not None:
            np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True

            best_loss = 1e9

        for ep in range(epochs):
            if (ep+1) % eval_per == 0:
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (ep+1) % eval_per == 0:
                preds = self.net.forward(X_test, inference=True)
                loss = self.net.loss.forward(preds, y_test)

                if early_stopping:
                    if loss < self.best_loss:
                        print(f"epoch : {ep+1} | Validation Loss : {loss:.4f}")
                        self.best_loss = loss
                    else:
                        print(f"loss increased | taking model with loss : {self.best_loss:.4f} of the epoch {ep+1-eval_per}")
                        self.net = last_model

                        setattr(self.optim, 'net', self.net)
                        break

                else:
                    print(f"epoch : {ep+1} Validation Loss : {loss: 0.4f}")

            if self.optim.finalr:
                self.optim._decay_lr()
