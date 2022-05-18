#import torch
from torch import empty, manual_seed
import random, math
random.seed(0)
manual_seed(0)

def MSE_func(v, t):
    return (v - t).pow(2).sum()

def dMSE(v, t):
    return 2 * (v - t)

def ReLU_func(x):
    x_ = x.clone() # make an exact copy to allow safe manipulations of the image
    x_[x_<=0]=0
    return x_

def dReLU(x):
    x_ = x.clone()
    x_[x_<=0] = 0
    x_[x_>0] = 1
    return x_

def initialize(in_channels, out_channels, kernel_size):
    """
    Return weights and biases initialised such as the Conv2D Pytorch documentation 
    Input:
        * in_channels (int) - Number of channels in the input image
        * out_channels (int) - Number of channels produced by the convolution
        * kernel_size (int or tuple) - Size of the convolving kernel
    Output: 
        * weights (tensor) - the learnable weights of the module initialized
        * biases (tensor) - the learnable biases of the module initialized
    """
    k = 1 / (in_channels * kernel_size[0] * kernel_size[1])

    weights = empty((out_channels, in_channels, kernel_size[0], kernel_size[1])).uniform_(-k**0.5, k**0.5)
    biases = empty(out_channels).uniform_(-k**0.5, k**0.5)

    return weights, biases


def Sigmoid_func(x):
    x_ = x.clone()
    return 1 / (1 + 1/x_.exp())
    
def dSigmoid(x):
    x_ = x.clone()
    return Sigmoid_func(x_) * (1 - Sigmoid_func(x_))
