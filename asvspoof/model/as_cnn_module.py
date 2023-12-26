from functools import partial
from typing import Optional, Callable, List, Union, Tuple

import torch
from torch import nn, Tensor
from torchvision.models._utils import _make_divisible
from torchvision.ops.misc import ConvNormActivation
from torchvision.utils import _log_api_usage_once


class ConvBNReLU1d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU1d, self).__init__(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU6(inplace=True)
        )


class Conv1dNormActivation(ConvNormActivation):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Optional[Union[int, Tuple[int, int], str]] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm1d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: Union[int, Tuple[int, int]] = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv1d,
        )


class InvertedResidual1d(nn.Module):
    def __init__(
            self, inp: int, oup: int, stride: int = 1, expand_ratio: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv1dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv1dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SqueezeExcitation1d(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            groups: int = 1,
            activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
            scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Conv1d(input_channels, squeeze_channels, 1, groups=groups)
        self.fc2 = torch.nn.Conv1d(squeeze_channels, input_channels, 1, groups=groups)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidual1dSE(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
            # self,
            # cnf: InvertedResidualConfig,
            # norm_layer: Callable[..., nn.Module],
            # se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation1d, scale_activation=nn.Hardsigmoid),

            self,
            input_channels: int,
            out_channels: int,
            expanded_channels: int,
            kernel_size: int = 3,
            stride: int = 1, expand_ratio: int = 1,
            dilation: int = 1, use_hs=False, use_se=True,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation1d, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if use_hs else nn.ReLU

        # expand
        if expanded_channels != input_channels:
            layers.append(
                Conv1dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    groups=groups,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if dilation > 1 else stride
        layers.append(
            Conv1dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if use_se:
            squeeze_channels = _make_divisible(expanded_channels // 4, 8)
            layers.append(se_layer(input_channels=expanded_channels, squeeze_channels=squeeze_channels, groups=groups))

        # project
        layers.append(
            Conv1dNormActivation(
                expanded_channels, out_channels, kernel_size=1, groups=groups, norm_layer=norm_layer,
                activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels
        self._is_cn = stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result
