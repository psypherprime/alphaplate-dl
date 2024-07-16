import numpy as np

from activations import Linear, Sigmoid, Dropout, Flatten
from core import Operation, ParamOperation
from layerops import WeightMul, BiasAdd, Conv2D_Op

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
        raise NotImplementedError()

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

        # for operation in self.operations[::-1]:
        for operation in reversed(self.operations):
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

    def __init__(self, neurons: int, activation: Operation = Sigmoid(), conv_in: bool = False,
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


class Conv2D(Layer):

    def __init__(self, out_channels: int, param_size: int, dropout: float = 1.0,
                 weight_init: str = 'std', activation: Operation = Linear(),
                 flatten: bool = False) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.out_channels = out_channels
        self.dropout = dropout
        self.flatten = flatten
        self.weight_init = weight_init
        self.activation = activation

    def _setup_layer(self, input_: np.ndarray) -> None:
        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == 'glorot':
            scale = 2.0 / (in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(loc=0, scale=scale, size=(self.input_.shape[1], in_channels, self.param_size, self.param_size))
        self.params.append(conv_param)

        self.operations = [Conv2D_Op(conv_param), self.activation]

        if self.flatten:
            self.operations.append(Flatten())

        if self.dropout:
            self.operations.append(Dropout(self.dropout))

        return None
