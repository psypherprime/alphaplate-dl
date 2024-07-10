import numpy as np


class Operation(object):
    """
    adam class for all operations
    in the neural network
    """

    def __init__(self):
        self.input_ = None
        self.output = None
        self.input_grad = None

    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:
        self.input_ = input_
        self.output = self._output(inference)

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        backward layer for the input gradients
        """

        assert self.output.shape == output_grad.shape
        self.input_grad = self._input_grad(output_grad)
        assert self.input_.shape == self.input_grad.shape

        return self.input_grad

    def _output(self, inference: bool = False) -> np.ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ParamOperation(Operation):
    """
    a parent parameter class for weights
    and biases of the layers
    """

    def __init__(self, param: np.ndarray):
        super().__init__()
        self.param = param
        self.param_grad = None

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        assert self.output.shape == output_grad.shape
        # self.input_grad = self._input_grad(output_grad) # Why the repeated implementation
        self.param_grad = self._param_grad(output_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
