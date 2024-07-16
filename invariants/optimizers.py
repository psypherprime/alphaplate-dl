import numpy as np

"""
implement other optimizers as well
SGDM, Adam, AdaGrad, RMSProp
"""


class Optimizer(object):

    def __init__(self, lr: float = 0.01, finalr: float = 1e9, decay_type: str = None):
        self.lr = lr
        self.finalr = finalr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self):
        if self.decay_type is None:
            return
        elif self.decay_type == "exponential":
            self.decay_rate = np.power(self.finalr / self.lr, 1.0 / (self.max_epochs - 1))
        elif self.decay_type == "linear":
            self.decay_rate = (self.lr - self.finalr) / (self.max_epochs - 1)

    def _decay_lr(self) -> None:
        if not self.decay_type:
            return
        elif self.decay_type == "exponential":
            self.lr *= self.decay_rate
        elif self.decay_type == "linear":
            self.lr -= self.decay_rate

    def step(self, epoch: int = 0) -> None:

        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param=param, param_grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()


class SGD(Optimizer):

    def __init__(self, lr: float = 0.01, finalr: float = -1e9, decay_type: str = None):
        super().__init__(lr, finalr, decay_type)

    def _update_rule(self, **kwargs) -> None:
        update = self.lr * kwargs['param_grad']
        kwargs['param'] -= update
