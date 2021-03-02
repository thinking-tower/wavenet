import torch
import torch.nn as nn

class CausalConvolution1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        super().__init__()
        self.dilated_convolution = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
    def forward(self, x):
        out = self.dilated_convolution(x)
        # TODO:
        # 1. Shift output forward by one so that prediction for timestep t uses only up to the t-1 sample