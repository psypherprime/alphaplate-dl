import numpy as np
from copy import deepcopy
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Operation(object):
    '''
    Base class for all operations in a
    neural network.

    Created since the other operations classes are
    going to inherit from this class.
    '''
    def __init__(self):
        pass

    def forward(self, input_:np.ndarray):
        self.input_ = input_
        self.output = self._output()

        return self.output

    def backward(self, output_grad:np.ndarray) -> np.ndarray:

        assert self.output.shape == output_grad.shape
        self.input_grad = self._input_grad(output_grad) # Check this out

        assert self.input_.shape == self.input_grad.shape
        return self.input_grad

    def _output(self) -> np.ndarray:
        raise NotImplementedError

    def _input_grad(self, output_grad:np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ParamOperation(Operation):

    def __init__(self, param:np.ndarray) -> np.ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        assert self.output.shape == output_grad.shape

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert self.input_.shape == self.input_grad.shape
        assert self.param.shape == self.param_grad.shape

        return self.input_grad

    def _param_grad(self, output_grad:np.ndarray) -> np.ndarray:
        raise NotImplementedError


class WeightMultiply(ParamOperation):

    def __init__(self, W:np.ndarray):
        super().__init__(W)

    def _output(self) -> np.ndarray:
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad:np.ndarray) -> np.ndarray:
        return np.dot(output_grad, self.param.T)

    def _param_grad(self, output_grad:np.ndarray) -> np.ndarray: # Check this
        return np.dot(self.input_.T, output_grad)


class BiasAdd(ParamOperation):

    def __init__(self, B:np.ndarray):
        # assert B.shape[0]=1
        super().__init__(B)

    def _output(self) -> np.ndarray:
        return self.input_ + self.param

    def _input_grad(self, output_grad:np.ndarray) -> np.ndarray:
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad:np.ndarray) -> np.ndarray:
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])

class Sigmoid(Operation):

    def __init__(self):
        super().__init__()

    def _output(self) -> np.ndarray:
        return 1.0/(1+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad:np.ndarray) -> np.ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad

class Linear(Operation):
    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return self.input_

    def _input_grad(self, output_grad:np.ndarray) -> np.ndarray:
        return output_grad

class Layer(object):

    def __init__(self,neurons: int):
        self.neurons = neurons
        self.first = True
        self.params: list[np.ndarray] = []
        self.param_grads: list[np.ndarray] = []
        self.operations: list[Operation] = []


    def _setup_layer(self, num_in: int) -> None:
        raise NotImplementedError()

    def forward(self, input_: np.ndarray) -> np.ndarray:
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        assert self.output.shape == output_grad.shape

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._param_grads()

        return input_grad

    def _param_grads(self) -> np.ndarray:

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> np.ndarray:
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    def __init__(self, neurons: int, activation: Operation=Sigmoid()) -> None:
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: np.ndarray) -> None:
        if self.seed:
            np.random.seed(self.seed)

        self.params = []
        self.params.append(np.random.randn(input_.shape[1], self.neurons))
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None


class Loss(object):

    def __init__(self):
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        assert prediction.shape == target.shape

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> np.ndarray:

        self.input_grad = self._input_grad()
        assert self.prediction.shape == self.input_grad.shape

        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError

    def _input_grad(self) -> np.ndarray:
        raise NotImplementedError


class MeanSquaredError(Loss):

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> float:
        loss = np.sum(np.square(self.prediction - self.target))/self.prediction.shape[0]

        return loss

    def _input_grad(self) -> np.ndarray:
        return 2.0 * (self.prediction - self.target)/self.prediction.shape[0]


class NeuralNetwork(object):

    def __init__(self, layers: list[Layer], loss: Loss, seed:float = 1):
        self.layers = layers
        self.loss = loss
        self.seed = seed

        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: np.ndarray) -> np.ndarray:

        x_out = x_batch

        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: np.ndarray) -> None:

        grad = loss_grad

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:

        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):

        for layer in self.layers:
            yield from layer.params # Issue Here

    def param_grads(self):

        for layer in self.layers:
            yield from layer.param_grads

def mae(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def eval_regression_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray):
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))

class Optimizer(object):

    def __init__(self, lr: float = 1e-2):

        self.lr = lr

    def step(self) -> None:
        pass

class SGD(Optimizer):

    def __init__(self, lr: float = 1e-2) -> None:
        super().__init__(lr)

    def step(self):

        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad


class Trainer(object):

    def __init__(self, net: NeuralNetwork, optim: Optimizer):

        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def generate_batches(self, X: np.ndarray, y: np.ndarray, size:int=32) -> tuple[np.ndarray]:
        assert X.shape[0] == y.shape[0]

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch

    def fit(self, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray,
            epochs:int=100, eval_every:int=10, batch_size:int=32, seed:int=1, restart:bool=False):
        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for epoch in range(epochs):

            perm = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[perm], y_train[perm]

            if epoch % eval_every == 0:
                last_model = deepcopy(self.net)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

                if (epoch+1) % eval_every == 0:
                    test_preds = self.net.forward(X_test)
                    loss = self.net.loss.forward(test_preds, y_test)

                    if loss < self.best_loss:
                        print(f"Validation loss after {epoch+1} epochs: {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print(f"Loss increased after {epoch+1} epochs: {loss:.3f} final loss is {self.best_loss:.3f}")
                        self.net = last_model
                        setattr(self.optim, 'net', self.net)
                        break

def mk_2dnp(a: np.ndarray, type:str="col") -> np.ndarray:

    if type == "col":
        return a.reshape(-1, 1)
    else:
        return a.reshape(1, -1)

prp = NeuralNetwork(
    layers=[Dense(neurons=1,activation=Linear())],
    loss=MeanSquaredError(),
    seed=7
)

mlp = NeuralNetwork(
    layers=[Dense(neurons=16,activation=Sigmoid()),
            Dense(neurons=1,activation=Linear())],
    loss=MeanSquaredError(),
    seed=7
)

dnn = NeuralNetwork(
    layers=[Dense(neurons=16,activation=Sigmoid()),
            Dense(neurons=16,activation=Sigmoid()),
            Dense(neurons=1,activation=Linear())],
    loss=MeanSquaredError(),
    seed=7
)

dnn2 = NeuralNetwork(
    layers=[Dense(neurons=6,activation=Sigmoid()),
            Dense(neurons=6,activation=Sigmoid()),
            Dense(neurons=3,activation=Sigmoid()),
            Dense(neurons=1,activation=Linear())],
    loss=MeanSquaredError(),
    seed=7
)

'''
Clean this up
too much for bamboozlement
'''

# Regression Prediction
california = fetch_california_housing()
data, target = california.data, california.target
feature = california.feature_names

scaler = StandardScaler()
data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=7)
y_train, y_test = mk_2dnp(y_train), mk_2dnp(y_test)

trainer = Trainer(mlp, SGD(lr=0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs=100, eval_every=10)

# eval_regression_model(mlp, X_test, y_test)
#
# trainer = Trainer(dnn, SGD(lr=0.01))
# trainer.fit(X_train, y_train, X_test, y_test, epochs=100, eval_every=10)
#
# eval_regression_model(dnn, X_test, y_test)
#
# trainer = Trainer(prp, SGD(lr=0.001))
# trainer.fit(X_train, y_train, X_test, y_test, epochs=100, eval_every=10)
#
# eval_regression_model(dnn2, X_test, y_test)