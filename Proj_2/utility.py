#import torch
from torch import empty

import random, math


def ReLU(x):
    return max(0.0, x)

def df_ReLU(x):
    x = max(0.0, x)
    x[x>0] = 1
    return x

def weight_init(k, kernal_size):
    t = -2*math.sqrt(k) * torch.rand(kernal_size[0], kernal_size[1]) + math.sqrt(k)
    

