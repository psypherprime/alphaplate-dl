import numpy as np
from layer import Layer
from lossfunction import Loss, MeanSquaredError


class LayerBlock(object):

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, X_batch: np.ndarray, inference: bool = False):
        X_out = X_batch
        for layer in self.layers:
            X_out = layer.forward(X_out, inference)

        return X_out

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:

        grad = loss_grad
        for layer in self.layers:
            grad = layer.backward(grad)

        return grad

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads

    # What is this for
    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n " + ",\n ".join(layer_strs) + ")"


class NeuralNetwork(LayerBlock):

    def __init__(self, layers: list[Layer], loss: Loss = MeanSquaredError, seed: int = None):
        super().__init__(layers)
        self.loss = loss
        self.seed = seed

        if self.seed is not None:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward_loss(self, X_batch: np.ndarray, y_batch: np.ndarray, inference: bool = False) -> float:
        prediction = self.forward(X_batch, inference)
        return self.loss.forward(prediction, y_batch)

    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray, inference: bool = False) -> float:
        prediction = self.forward(X_batch, inference)

        batch_loss = self.loss.forward(prediction, y_batch)
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        return batch_loss
