import numpy as np
from core import ParamOperation

"""
Dense Layer
Class of Parametric Operations
"""


class WeightMul(ParamOperation):

    def __init__(self, W: np.ndarray):
        super().__init__(W)

    def _output(self, inference: bool = False) -> np.ndarray:
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.dot(output_grad, self.param.T)

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.dot(self.input_.T, output_grad)


class BiasAdd(ParamOperation):

    def __init__(self, B: np.ndarray):
        super().__init__(B)

    def _output(self, inference: bool = False) -> np.ndarray:
        return np.add(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        param_grad = np.ones_like(self.param)
        output_grad_reshaped = np.sum(output_grad, axis=0).reshape(1, -1)
        return param_grad * output_grad_reshaped