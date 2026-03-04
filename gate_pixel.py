import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
import py_func as pf


class MaskedConv3d(nn.Conv3d):
    """Regular Masked Convolutional Layer"""
    def __init__(self,*args, **kwargs):
        self.exclusive = kwargs.pop('exclusive')
        super(MaskedConv3d, self).__init__(*args, **kwargs)

        _, _, kh, kw, kd = self.weight.shape
        self.register_buffer('mask', torch.ones([kh, kw, kd]))
        self.mask[kh // 2, kw // 2,kd // 2 + (not self.exclusive):] = 0
        self.mask[kh // 2, kw // 2 + 1:,:] = 0
        self.mask[kh // 2 + 1:,:,:] = 0
        self.weight.data *= self.mask
        total_params = self.weight.numel()
        print("Total number of parameters in self.depth:", total_params)
        #self.weight.data = self.weight.data.to(torch.float32)
        #self.bias.data = self.bias.data.to(torch.float32)

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.conv3d(x, self.mask * self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups)
    

class ResidualMaskedConv3d(nn.Module):
    """Residual Masked Conv. layer"""
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            MaskedConv3d(input_dim, input_dim,kernel_size=5, padding=2,exclusive = False),nn.PReLU(),
        )

    def forward(self, x):
        return self.net(x) + x
    
class AutoregressiveColorPixelRCNN(nn.Module):
    """ Pixel Residual-CNN-class using residual blocks as shown in figure 5 from "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al. """
    def __init__(self, in_channels, out_channels, conv_filters):
        super().__init__()
        self.net = nn.Sequential(
            # A 7x7 A-type convolution
            MaskedConv3d(in_channels,conv_filters, kernel_size=5, padding=2,exclusive = True), nn.PReLU(),

            #type-B residual convolutons
            #ResidualMaskedConv3d(conv_filters), nn.PReLU(),
            ResidualMaskedConv3d(conv_filters), nn.PReLU(),
            ResidualMaskedConv3d(conv_filters), nn.PReLU(),
            ResidualMaskedConv3d(conv_filters), nn.PReLU(),
            ResidualMaskedConv3d(conv_filters), nn.PReLU(),
            ResidualMaskedConv3d(conv_filters), nn.PReLU(),
            ResidualMaskedConv3d(conv_filters), nn.PReLU(),
#            ResidualMaskedConv3d(conv_filters), nn.PReLU(),
            MaskedConv3d(conv_filters, out_channels, kernel_size=1, padding=0, exclusive = False),nn.PReLU(),
        )
        self.sig=nn.Sigmoid()
    
    def forward(self, x):
        y=(self.net(x))

        y=self.sig(y)
        return y
