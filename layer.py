import numpy as np

from activations import Linear, Sigmoid, Dropout  # Implement Dropout Function
from core import Operation, ParamOperation
from layerops import WeightMul, BiasAdd


class Layer(object):

    def __init__(self, neurons: int) -> None:
        self.neurons = neurons
        self.first = True
        self.params: list[np.ndarray] = []
        self.param_grads: list[np.ndarray] = []
        self.operations: list[Operation] = []
        self.input_ = None
        self.output = None

    def _setup_layer(self, input_: np.ndarray) -> None:
        pass

    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        assert output_grad.shape == self.output.shape

        for operation in self.operations[::-1]:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        assert self.input_.shape == input_grad.shape

        self._param_grads()

        return input_grad

    def _param_grads(self) -> None:
        self.param_grads = []

        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:
        self.params = []

        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):

    def __init__(self, neurons: int, activation: Operation = Linear(), conv_in: bool = False,
                 dropout: float = 1.0, weight_init: str = 'std') -> None:
        super().__init__(neurons)
        self.activation = activation
        self.dropout = dropout
        self.conv_in = conv_in
        self.weight_init = weight_init

    def _setup_layer(self, input_: np.ndarray) -> None:

        num_in = input_.shape[1]

        if self.weight_init == 'glorot':
            scale = 2.0 / (num_in + self.neurons)  # Check glorot
        else:
            scale = 1.0

        self.params = []
        # Weights initialization
        self.params.append(np.random.normal(loc=0, scale=scale, size=(num_in, self.neurons)))
        # Bias initialization
        self.params.append(np.random.normal(loc=0, scale=scale, size=(1, self.neurons)))

        self.operations = [WeightMul(self.params[0]), BiasAdd(self.params[1]), self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None
    