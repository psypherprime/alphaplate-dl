import numpy as np
from core import Operation



class Linear(Operation):

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return self.input

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad


class Sigmoid(Operation):

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backgrad = self.output * (1.0 - self.output)
        input_grad = output_grad * sigmoid_backgrad
        return input_grad


class Tanh(Operation):

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return np.tanh(self.input)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * (1 - (self.output * self.output)) # output of _output is stored in self.output


class ReLU(Operation):

    def __int__(self) -> None:
        super()._init__()

    def _output(self, inference: bool) -> np.ndarray:
        return np.where(self.input > 0, self.input, 0)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * (self.input >0)


class LeakyReLU(Operation):

    def __int__(self, revmul: float = 0.3) -> None:
        self.revmul = revmul
        super().__init__()

    def _output(self, inference: bool) -> np.ndarray:
        return np.where(self.input > 0, self.input, self.input*self.revmul)

    def _input_grads(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * np.where(self.input > 0, 1.0, self.revmul)
