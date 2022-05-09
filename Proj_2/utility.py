#import torch
from torch import empty, manual_seed
import random, math
random.seed(0)
manual_seed(0)

def MSE(v, t):
    return (v - t).pow(2).sum()

def dMSE(v, t):
    return 2 * (v - t)

def ReLU(x):
    return max(0.0, x)

def dReLU(x):
    x = max(0.0, x)
    x[x>0] = 1
    return x

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


def Simoid(x):
    return 1 / (1 + math.exp(-x))
    
def dSigmoid(x):
    return Simoid(x) * (1 - Simoid(x))
