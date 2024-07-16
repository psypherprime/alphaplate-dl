from neuralnetwork import NeuralNetwork
from train import Trainer
from layer import Dense
from activations import Linear, Sigmoid
from lossfunction import MeanSquaredError
from optimizers import SGD
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np


def mk_2dnp(a: np.ndarray, type: str = "col") -> np.ndarray:
    if type == "col":
        return a.reshape(-1, 1)
    else:
        return a.reshape(1, -1)


# Loading Dataset - California housing toy dataset
California = fetch_california_housing()
data, target = California.data, California.target
features = California.feature_names

scaler = StandardScaler()
data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
y_train, y_test = mk_2dnp(y_train), mk_2dnp(y_test)

mlp = NeuralNetwork(
    layers=[Dense(neurons=16,activation=Sigmoid()),
            Dense(neurons=256,activation=Sigmoid()),
            Dense(neurons=64,activation=Sigmoid()),
            Dense(neurons=4,activation=Sigmoid()),
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

trainer = Trainer(mlp, SGD(lr=0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs=1000, eval_per=10)


def eval_regression_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray):
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))


def mae(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


eval_regression_model(mlp, X_test, y_test)
preds = mlp.forward(X_test)
for i, (p, y) in enumerate(zip(preds, y_test)):
    print(p, y)