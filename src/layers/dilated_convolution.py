import torch
import torch.nn as nn

class CausalDilatedConvolution1d(nn.Module):
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
        self.kernel_size = self.dilated_convolution.weight.shape[-1]

    def forward(self, input_):
        """
        Consider the following output for a non-causal convolution:

        output:             1 2
                           /|/|
        hidden_layer:     1 2 3  
                         /|/|/|
        input_:         1 2 3 4

        For output_2 to not see input_4 (the current timestep we're trying to predict), 
        we should shift the input to hidden layer forward by 1.  

        output:             1 2
                           /|/|
        hidden_layer:       1 2 3 (3 is discarded)
                           / / /
                          | | |
                         /|/|/|
        input_:         1 2 3 4
        Now both output_2 and output_1 can't see the input at the current timestep but can see all before it.
        See resources/WO2018048934A1.pdf, resources/iclr2017.pdf, resources/iclr2017.pdf.

        Args:
            input_ (torch.Tensor): (Batches, Channels, Width)

        Returns:
            torch.Tensor: (Batches, Channels, Width)
        """
        output = self.dilated_convolution(input_)
        return output[:, :, -(self.kernel_size-1)]