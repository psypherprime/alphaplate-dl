import numpy as np

class Operation(object):

    def __init__(self):

        self.input_ = None
        self.output = None
        self.input_grad = None

    def forward(self, input_: np.ndarray, inference: bool=False) -> np.ndarray:

        self.input_ = input_
        self.output = _output(inference)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        assert self.output.shape == output_grad.shape
        self.input_grad = self._input_grad(output_grad)
        assert self.input_.shape == self.input_grad.shape

        return self.input_grad

    def _output(self, inference: bool=False) -> np.ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()