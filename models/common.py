import torch
import torch.nn as nn


def autopad(k, p=None):  # kernel, padding, dilation
    """Automatically calculates same shape padding for convolutional layers, optionally adjusts for dilation."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """A standard Conv2D layer with batch normalization and optional activation for neural networks."""

    def __init__(self, c1, c2, k=1, s=1, p=None):
        """Initializes a standard Conv2D layer with batch normalization and optional activation; args are channel_in,
        channel_out, kernel_size, stride, padding, groups, dilation, and activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        """Applies convolution, batch normalization, and activation to input `x`; `x` shape: [N, C_in, H, W] -> [N,
        C_out, H_out, W_out].
        """
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Implements a bottleneck layer with optional shortcut for efficient feature extraction in neural networks."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        """Initializes a standard bottleneck layer with optional shortcut; args: input channels (c1), output channels
        (c2), shortcut (bool), groups (g), expansion factor (e).
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Executes forward pass, performing convolutional ops and optional shortcut addition; expects input tensor
        x.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Concat(nn.Module):
    """Concatenates a list of tensors along a specified dimension for efficient feature aggregation."""

    def __init__(self, dimension=1):
        """Initializes a module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; x is a list of tensors to concatenate, dimension
        defaults to 1.
        """
        return torch.cat(x, self.d)

