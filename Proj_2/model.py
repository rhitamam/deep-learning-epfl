from torch import empty , cat , arange, Tensor
from torch.nn.functional import fold, unfold
from utility import *
import random, math
torch.set_grad_enabled(False)


class Module(object) :
    def forward (self,  input) :
        raise NotImplementedError
    def backward (self,  gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []

class Conv2d(Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, device=None):
        self.in_chan = in_channels
        self.out_chan = out_channels
        self.k_size = kernel_size
        self.str = stride
        self.pad = padding
        self.dil = dilation
        self.dev = device
        self.weights = empty(in_channels, out_channels) 
        self.bias = empty(out_channels)
        self.w_grad = empty(self.weights.shape)
        self.b_grad = empty(self.bias.shape)
        self.input = Tensor()
    
    def forward (self,  input) :
        #add padd stride and everything
        self.input = input
        linear = (input @ self.weights) + self.bias
        unfolded = unfold(linear, kernel_size= self.kernel_size)
        wxb = self.weights.view(self.out_chan, -1) @ unfolded + self.bias.view(1, -1, 1)
        output = wxb.view(1, self.out_chan, linear.shape[2] - self.k_size[0] + 1, linear.shape[3] - self.k_size[1]+ 1)
        return output
        
    def backward (self,  gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        """
        Return a list of pairs composed of a parameter tensor and a gradient tensor of the same size
        Output: 
              list of pairs composed of a parameter tensor and a gradient tensor
        """
        return [(self.weights, self.w_grad), (self.bias, self.b_grad)]


class MSE (object) :
    def __init__():
        self.tensor = empty()
        self.target = empty()

    def forward (self,  input, target):
        self.tensor = input
        self.target = target
        return MSE(input, target)
    def backward (self, gradwrtoutput):
        return dMSE(input, target) 
    def param (self) :
        return []


class ReLU (object) :
    def __init__(self):
        self.tensor = empty()

    def forward (self,  input):
        self.tensor = input
        return ReLU(input)

    def backward (self, gradwrtoutput):
        return gradwrtoutput * dReLU(self.tensor)

    def param (self) :
        return []


class Sigmoid (object) :
    def forward (self,  input) :
        raise NotImplementedError
    def backward (self, gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []

class SGD (object) :
    def forward (self,  input) :
        raise NotImplementedError
    def backward (self,  gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []

class Sequential (object) :
    def __init__(self, modules):
        self.modules = modules

    def forward (self,  input) :
        self.input = input
        for module in self.modules:
            input = module.forward(input)
        return input
    def backward (self, gradwrtoutput):
        self.gradwrtoutput = gradwrtoutput
        for module in self.modules:
            gradwrtoutput = module.forward(gradwrtoutput)
        return gradwrtoutput
    def param (self) :
        return []

'''
Conv2d, TransposeConv2d or NearestUpsampling, ReLU, Sigmoid, MSE, SGD, Sequential.
• Convolution layer.
• Transpose convolution layer, or alternatively a combination of Nearest neighbor upsampling + Convolution.
• Upsampling layer, which is usually implemented with transposed convolution, but you can alternatively use a combination of Nearest neighbor upsampling + Convolution for this mini-project.
• ReLU
• Sigmoid
• A container like torch.nn.Sequential to put together an arbitrary configuration of modules together.
• Mean Squared Error as a Loss Function
• Stochastic Gradient Descent (SGD) optimizer
'''