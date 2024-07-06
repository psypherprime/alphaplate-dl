import numpy as np


def normalize(a: np.ndarray) -> np.ndarray:
    b = 1 - a
    return np.concatenate([a, b], axis=1)


def denormalize(a: np.ndarray):
    return a[np.newaxis, 0]


class Loss(object):

    def __init__(self):
        self.predictions = None
        self.targets = None
        self.output = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        assert predictions.shape == targets.shape
        self.predictions = predictions
        self.targets = targets

        self.output = self._output

        return self.output

    def backward(self) -> np.ndarray:
        self.input_grad = self._input_grad()

        assert self.predictions.shape == self.input_grad.shape

        return self.input_grad

    def _output(self) -> np.ndarray:
        raise NotImplementedError()

    def _input_grad(self) -> np.ndarray:
        raise NotImplementedError()


class MeanSquaredError(Loss):

    def __init__(self):
        super().__init__()
        # check for normalization of mse

    def _output(self) -> float:
        return (np.sum(np.square(self.predictions - self.targets))) / len(self.targets)

    def _input_grad(self) -> np.ndarray:
        return 2.0 * (self.predictions - self.targets) / len(self.targets)


class SoftmaxCrossEntropy(Loss):

    """
    Implement this function
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> np.ndarray:

        if self.predictions.shape[0] == 1:
            self.single_class = True

        if self.single_class:
            self.predictions, self.targets = normalize(self.predictions), normalize(self.targets)

        raise NotImplementedError()

    def _input_grad(self) -> np.ndarray:
        # if self.single_class:
            # return denormalize(self.preds - self.targets)

        raise NotImplementedError()