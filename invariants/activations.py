import numpy as np
from core import Operation


class Linear(Operation):
    """
    Linear Activation Function
    works as
    for all x : x = x

    range : (-inf, inf)
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool = False) -> np.ndarray:
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad


class Sigmoid(Operation):
    """
    Sigmoid Activation Function
    works as
    for all x : x = 1/(1+exp(-x))

    range : (0, 1)
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool = False) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backgrad = self.output * (1.0 - self.output)
        input_grad = output_grad * sigmoid_backgrad
        return input_grad


class Tanh(Operation):

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool = False) -> np.ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * (1 - (self.output * self.output))  # output of _output is stored in self.output


class ReLU(Operation):
    """
    ReLU Function
    works as
    for x >= 0 : x = x
    for x < 0 : x = 0

    range : [0, inf)
    """

    def __int__(self) -> None:
        super().__init__()

    def _output(self, inference: bool = False) -> np.ndarray:
        return np.where(self.input_ > 0, self.input_, 0)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * (self.input_ > 0)


class LeakyReLU(Operation):
    """
    LeakyReLU Function
    works as
    for x >= 0 : x = x
    for x < 0 : x = -x*negmul

    range : (-inf, inf)
    """

    def __int__(self, negmul: float = 0.3) -> None:
        self.negmul = negmul
        super().__init__()

    def _output(self, inference: bool = False) -> np.ndarray:
        return np.where(self.input_ > 0, self.input_, self.input_ * self.negmul)

    def _input_grads(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * np.where(self.input_ > 0, 1.0, self.negmul)


class Dropout(Operation):

    def __init__(self, probability: float = 0.5) -> None:
        super().__init__()
        self.probability = probability

    def _output(self, inference: bool = False) -> np.ndarray:
        if inference:
            return self.input_ * self.probability
        else:
            self.mask = np.random.binomial(1, self.probability, size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grads(self, output_grad: np.ndarray) -> np.ndarray:
        return self.input_ * self.mask


class Flatten(Operation):

    def __init__(self):
        super().__init__()

    def _output(self, inference: bool = False) -> np.ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grads(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad.reshape(self.input_.shape)