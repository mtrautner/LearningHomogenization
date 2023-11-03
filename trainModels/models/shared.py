'''
Adapted from https://github.com/nickhnelsen/FourierNeuralMappings 
'''

import torch
import torch.nn as nn
import torch.fft as fft
import math


def compl_mul(input_tensor, weights):
    """
    Complex multiplication:
    (batch, in_channel, ...), (in_channel, out_channel, ...) -> (batch, out_channel, ...), where ``...'' represents the spatial part of the input.
    """
    return torch.einsum("bi...,io...->bo...", input_tensor, weights)


################################################################
#
# 2d helpers
#
################################################################
def resize_rfft2(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft2(ar, s=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N_1, N_2) tensor, must satisfy real conjugate symmetry (not checked)
        s: (2) tuple, s=(s_1, s_2) desired irfft2 output dimension (s_i >=1)
    Output
        out: (..., s1, s_2//2 + 1) tensor
    """
    s1, s2 = s
    out = resize_rfft(ar, s2) # last axis (rfft)
    return resize_fft(out.transpose(-2,-1), s1).transpose(-2,-1) # second to last axis (fft)

    
def get_grid2d(shape, device):
    """
    Returns a discretization of the 2D identity function on [0,1]^2
    """
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)

def get_grid2d_torus(shape, device):
    """
    Returns a discretization of the 2D identity function on \T^2
    """
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    gridxtor = torch.sin(2*math.pi*gridx)
    gridxtorshift = torch.sin(2*math.pi*(gridx + 0.1)) # makes physical info unique
    gridytor = torch.sin(2*math.pi*gridy)
    gridytorshift = torch.sin(2*math.pi*(gridy + 0.1))
    return torch.cat((gridxtor, gridytor), dim=-1).to(device)


def projector2d(x, s=None):
    """
    Either truncate or zero pad the Fourier modes of x so that x has new resolution s (s is 2 tuple)
    """
    if s is not None and tuple(s) != tuple(x.shape[-2:]):
        x = fft.irfft2(resize_rfft2(fft.rfft2(x, norm="forward"), s), s=s, norm="forward")
        
    return x


################################################################
#
# 2d Fourier layers
#
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier integral operator layer defined for functions over the torus. Maps functions to functions.
        """
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, s=None):
        """
        Input shape (of x):     (batch, channels, nx_in, ny_in)
        s:                      (list or tuple, length 2): desired spatial resolution (s,s) in output space
        """
        # Original resolution
        xsize = x.shape[-2:]
        
        # Compute Fourier coeffcients (un-scaled)
        x = fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x.shape[0], self.out_channels, xsize[-2], xsize[-1]//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes1, :self.modes2] = \
            compl_mul(x[..., :self.modes1, :self.modes2], self.weights1)
        out_ft[..., -self.modes1:, :self.modes2] = \
            compl_mul(x[..., -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if s is None or tuple(s) == tuple(xsize):
            x = fft.irfft2(out_ft, s=tuple(xsize))
        else:
            x = fft.irfft2(resize_rfft2(out_ft, s), s=s, norm="forward") / (xsize[-2] * xsize[-1])

        return x


class LinearFunctionals2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier neural functionals (encoder) layer for functions over the torus. Maps functions to vectors.
        Inputs:    
            in_channels  (int): number of input functions
            out_channels (int): total number of linear functionals to extract
        """
        super(LinearFunctionals2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
    
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2
    
        # Complex conjugation in L^2 inner product is absorbed into parametrization
        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, 2*self.modes1, self.modes2 + 1, dtype=torch.cfloat))

    def forward(self, x):
        """
        Input shape (of x):     (batch, in_channels, ..., nx_in, ny_in)
        Output shape:           (batch, out_channels, ...)
        """
        # Compute Fourier coeffcients (scaled to approximate integration)
        x = fft.rfft2(x, norm="forward")
        
        # Truncate input modes
        x = resize_rfft2(x, (2*self.modes1, 2*self.modes2))

        # Multiply relevant Fourier modes and take the real part
        x = compl_mul(x, self.weights).real

        # Integrate the conjugate product in physical space by summing Fourier coefficients
        x = 2*torch.sum(x[..., :self.modes1, :], dim=(-2, -1)) + \
            2*torch.sum(x[..., -self.modes1:, 1:], dim=(-2, -1)) - x[..., 0, 0]

        return x
    

class LinearDecoder2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier neural decoder layer for functions over the torus. Maps vectors to functions.
        Inputs:    
            in_channels  (int): dimension of input vectors
            out_channels (int): total number of functions to extract
        """
        super(LinearDecoder2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, 2*self.modes1, self.modes2 + 1, dtype=torch.cfloat))

    def forward(self, x, s):
        """
        Input shape (of x):             (batch, in_channels, ...)
        s (list or tuple, length 2):    desired spatial resolution (nx,ny) of functions
        Output shape:                   (batch, out_channels, ..., nx, ny)
        """
        # Multiply relevant Fourier modes
        x = compl_mul(x[...,None,None].type(torch.cfloat), self.weights)
        
        # Zero pad modes
        x = resize_rfft2(x, tuple(s))
        
        # Return to physical space
        return fft.irfft2(x, s=s)
