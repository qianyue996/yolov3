import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Automatically calculates same shape padding for convolutional layers, optionally adjusts for dilation."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """A standard Conv2D layer with batch normalization and optional activation for neural networks."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard Conv2D layer with batch normalization and optional activation; args are channel_in,
        channel_out, kernel_size, stride, padding, groups, dilation, and activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies convolution, batch normalization, and activation to input `x`; `x` shape: [N, C_in, H, W] -> [N,
        C_out, H_out, W_out].
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies fused convolution and activation to input `x`; input shape: [N, C_in, H, W] -> [N, C_out, H_out,
        W_out].
        """
        return self.act(self.conv(x))

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

class Contract(nn.Module):
    """Contracts spatial dimensions into channels, e.g., (1,64,80,80) to (1,256,40,40) with a specified gain."""

    def __init__(self, gain=2):
        """Initializes Contract module to refine input dimensions, e.g., from (1,64,80,80) to (1,256,40,40) with a
        default gain of 2.
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor (b,c,h,w) to contracted shape (b,c*s^2,h/s,w/s) with default gain s=2, e.g.,
        (1,64,80,80) to (1,256,40,40).
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

class Expand(nn.Module):
    """Expands spatial dimensions of input tensor by a factor while reducing channels correspondingly."""

    def __init__(self, gain=2):
        """Initializes Expand module to increase spatial dimensions by factor `gain` while reducing channels
        correspondingly.
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Expands spatial dimensions of input tensor `x` by factor `gain` while reducing channels, transforming shape
        `(B,C,H,W)` to `(B,C/gain^2,H*gain,W*gain)`.
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)