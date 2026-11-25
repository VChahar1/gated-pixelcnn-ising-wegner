import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import sys


class MaskedConv3d(nn.Module):
    """
    Two-stack masked convolution with gated activations and a residual-connection

    See figure 2 in _Conditional Image Generation with PixelCNN Decoders_, van den Oord 2016.
    """

    def __init__(
        self,
        channels,
        colors,
        skip_connection=True,
        res_connection=True,
        gates=False,
        self_connection=False,
        self_start=True,
        k=7,
        padding=3,
    ):
        ':param self_connection: Whether to mask out the "current pixel" (ie. the middle of the convolution). In the first layer, this should be masked out, since it connects to the value we\'re trying to predict. In higher layers'
        super().__init__()
        assert (k // 2) * 2 == k - 1

        self.res_connection = res_connection
        self.gates = gates
        self.skip_connection = skip_connection
        f = colors if self_start else channels
        print("kernel", k, f, channels)
        self.depth = nn.Conv3d(f, channels, kernel_size=k, padding=padding, bias=False)
        self.vertical = nn.Conv3d(
            f, channels, kernel_size=(1, k, k), padding=(0, padding, padding), bias=True
        )
        self.horizontal = nn.Conv3d(
            f, channels, kernel_size=(1, 1, k), padding=(0, 0, padding), bias=True
        )

        total_params = sum(p.numel() for p in self.vertical.parameters())
        print("Total number of parameters in self.depth:", total_params)

        self.register_buffer("dmask", self.depth.weight.data.clone())
        self.register_buffer("vmask", self.vertical.weight.data.clone())
        self.register_buffer("hmask", self.horizontal.weight.data.clone())

        self.dmask.fill_(1)
        self.vmask.fill_(1)
        self.hmask.fill_(1)

        self.dmask[:, :, k // 2 + 1 :, :, :] = 0

        self.vmask[:, :, :, k // 2 + 1 :, :] = 0

        self.hmask[:, :, :, :, k // 2 + 1 :] = 0

        m = k // 2

        if self_connection:
            self.hmask[:, :, 0, 0, m] = 0
            self.vmask[:, :, :, k // 2, :] = 0
            self.dmask[:, :, k // 2 :, :, :] = 0

    def forward(self, x):

        dxin, vxin, hxin, skip = x
        self.depth.weight.data *= self.dmask
        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        dx = self.depth.forward(dxin)
        vx = self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        vx_1 = vx + dx

        hx = F.elu(hx)

        return F.elu(dx), F.elu(vx_1), hx, skip


class Gated_pixel(nn.Module):
    """
    Model combining several gated pixelCNN layers
    """

    def __init__(self, input_size, channels, num_layers, k=7, padding=3):
        super().__init__()

        bs, c, d, h, w = input_size

        self.conv1 = nn.Conv3d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            print(i)
            self.gated_layers.append(
                MaskedConv3d(
                    channels,
                    colors=c,
                    skip_connection=i > 0,
                    res_connection=i > 0,
                    gates=False,
                    self_connection=(i <= 0),
                    self_start=False,
                    k=k,
                    padding=padding,
                )
            )

        self.conv2 = nn.Conv3d(channels, c, 1, groups=c)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        b, c, d, h, w = x.size()
        cout = 0
        x = self.conv1((x))
        xd, xv, xh, cout = x, x, x, cout

        for layer in self.gated_layers:
            xd, xv, xh, cout = layer((xd, xv, xh, cout))
        y = self.conv2(F.elu(xh))
        y = self.sig(y)
        return y
