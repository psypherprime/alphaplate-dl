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


class Conv2D_Op(ParamOperation):

    def __init__(self, params: np.ndarray):
        super().__init__(params)
        self.param_size = params.shape[2]
        self.param_pad = self.param_size // 2

    def _pad_1d(self, input_: np.ndarray) -> np.ndarray:
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, input_, z])

    def _pad_1d_batch(self, input_: np.ndarray) -> np.ndarray:
        out = [self._pad_1d(inp) for inp in input_]
        return np.stack(out)

    def _pad_2d_obs(self, input_: np.ndarray) -> np.ndarray:
        inp_pad = self._pad_1d_batch(input_)
        oth = np.zeros((self.param_pad, input_.shape[0] + self.param_pad * 2))
        return np.concatenate([oth, inp_pad, oth])

    def _pad_2d_channel(self, input_: np.ndarray) -> np.ndarray:
        return np.stack([self._pad_2d_obs(chn) for chn in input_])

    def _get_image_patches(self, input_: np.ndarray) -> np.ndarray:
        imgs_pad_bat = np.stack([self._pad_2d_channel(obs) for obs in input_])
        patches = []
        img_height = imgs_pad_bat.shape[2]

        for h in range(img_height - self.param_size + 1):
            for w in range(img_height - self.param_size + 1):
                patch = imgs_pad_bat[:, :, h:h + self.param_size, w:w + self.param_size]
                patches.append(patch)

        return np.stack(patches)

    def _output(self, inference: bool = False) -> np.ndarray:

        batch_size = self.input_.shape[0]
        img_height = self.input_.shape[2]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]

        patches = self._get_image_patches(self.input_)

        patches_reshaped = (patches.transpose(1, 0, 2, 3, 4).reshape(batch_size, img_size, -1))

        param_reshaped = (self.param.transpose(0, 2, 3, 1).reshape(batch_size, -1))

        output_reshaped = (np.matmul(patches_reshaped, param_reshaped)
                           .reshape(batch_size, img_height, img_height, -1)
                           .transpose(0, 3, 1, 2))

        return output_reshaped

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        output_patches = (self._get_image_patches(output_grad)
                          .transpose(1, 0, 2, 3, 4)
                          .reshape(batch_size, img_size, -1))

        param_reshaped = (self.param
                          .reshape(self.param.shape[0], -1)
                          .T)

        return (np.matmul(output_patches, param_reshaped)
                .reshape(batch_size, img_height, img_height, self.param.shape[0])
                .transpose(0, 3, 1, 2))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        in_patches_reshaped = (self._get_image_patches(self.input_)
                      .reshape(batch_size, img_size, -1)
                      .T)

        out_grad_reshaped = (output_grad
                            .transpose(0, 2, 3, 1)
                            .reshape(batch_size * img_size, -1))

        return (np.matmul(in_patches_reshaped, out_grad_reshaped)
                .reshape(in_channels, self.param_size, self.param_size, out_channels)
                .transpose(0, 3, 1, 2))
