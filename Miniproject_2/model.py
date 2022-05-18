from torch import empty , cat , arange, Tensor, manual_seed, repeat_interleave
from torch.nn.functional import fold, unfold
from others.otherfile1 import *
import random, math
random.seed(0)
manual_seed(0)
#torch.set_grad_enabled(False)


class Module(object) :
    def forward (self,  input) :
        raise NotImplementedError
    def backward (self,  gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        self.in_chan = in_channels
        self.out_chan = out_channels
        if (isinstance(kernel_size, int)):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weights, self.bias = initialize(in_channels, out_channels, self.kernel_size)
        self.w_grad = empty(self.weights.shape)
        self.b_grad = empty(self.bias.shape)
        self.input = Tensor()
    
    def forward (self,  input) :
        #add padd stride and everything
        self.input = input
        unfolded = unfold(input, kernel_size=self.kernel_size)
        wxb = self.weights.view(self.out_chan, -1) @ unfolded + self.bias.view(1, -1, 1)
        output = wxb.view(input.shape[0], self.out_chan, input.shape[2] - self.kernel_size[0] + 1, input.shape[3] - self.kernel_size[1]+ 1)
        '''
        add 0 zeros to the input
        add stride 
        unfolded = unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        output = wxb.view(input.shape[0], 
                            self.out_chan,
                            (1+(input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0]).floor_(),
                            (1+(input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1]).floor_())
        '''
        return output
        
    def backward (self,  gradwrtoutput):
        'test wrt to grad of conv2d !!'
        self.b_grad = gradwrtoutput
        
        fold = fold(output_size=self.w_grad.shape, kernel_size=self.kernel_size)
        self.w_grad = gradwrtoutput @ fold(self.input)
        return (self.w_grad, self.b_grad)

    def param (self):
        """
        Return a list of pairs composed of a parameter tensor and a gradient tensor of the same size
        Output: 
            * list of pairs composed of a parameter tensor and a gradient tensor
        """
        return [(self.weights, self.w_grad), (self.bias, self.b_grad)]

class NeerestUpSampling(Module) :
    def __init__(self,scalefactor):
        self.scalefactor = scalefactor
        
    
    def forward(self, input):
        first = repeat_interleave(input, 2, dim=3)
        final =  repeat_interleave(first,2,dim=2)
        return final

    def backward(self, gradwrtoutput):
         self.gradwrtoutput = gradwrtoutput
         #raise NotImplementedError
         
class SGD(object) :
    def __init__(self, module, lr=0.1, momentum=0.9):
        self.module = module
        self.lr = lr
        self.momentum = momentum

    def update (self):
        weights, w_grad, bias, b_grad = self.module.params()
        weights -= self.lr * w_grad
        bias -= self.lr * b_grad
        return (weights, bias)


class Sequential(Module) :
    def __init__(self, modules):
        self.modules = modules
        self.target = Tensor()
        self.input = Tensor()
        self.loss = MSE()

    def forward (self, input, target) :
        self.input = input
        self.target = target
        for module in self.modules:#
            input = module.forward(input)
        self.loss.forward(input, self.target)
        return input
    
    def backward (self):
        gradwrtoutput = self.loss.backward()
        for module in reversed(self.modules):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput
    
    def param (self) :
        'should return multiple params'
        param = [m.param() for m in self.modules]
        return param

class MSE(Module) :
    def __init__(self):
        self.tensor = Tensor()
        self.target = Tensor()

    def forward (self,  input, target):
        self.tensor = input
        self.target = target
        return MSE_func(input, target)
    def backward (self):
        return dMSE(self.tensor, self.target) 
    def param (self) :
        return []


class ReLU(Module) :
    def __init__(self):
        self.tensor = Tensor()

    def forward (self,  input):
        self.tensor = input
        print(self.tensor)
        return ReLU_func(input)

    def backward (self, gradwrtoutput):
        return gradwrtoutput * dReLU(self.tensor)

    def param (self) :
        return []


class Sigmoid(Module) :
    def __init__(self):
        self.tensor = Tensor()

    def forward (self,  input):
        self.tensor = input
        return Sigmoid_func(input)

    def backward (self, gradwrtoutput):
        return gradwrtoutput * dSigmoid(self.tensor)

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

'''
report:
- plot 
- partie théorique
- réutilisation de l'intro ?
- rapport 2 individuels marquer la séparation ?
'''

'''
#==============================================================================
from charset_normalizer import from_path
import torch
from model_copy import *
from torch.nn import functional
random.seed(0)
torch.manual_seed(0)

in_channels = 3 
out_channels = 4
kernel_size = (2, 3)

x = torch.randn((1, in_channels , 32, 32))
y = torch.randn((1, out_channels , 31, 30))

import torch.nn as nn

model = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size),
          nn.ReLU(),
          nn.Sigmoid()
        )

#print(model.parameters())

seq = Sequential([Conv2d(in_channels, out_channels, kernel_size), ReLU(), Sigmoid()])
seq.forward(x,y)
seq.backward()
'''