#import torch
from torch import empty
import math

def ReLU(x):
    return max(0.0, x)

def df_ReLU(x):
    x = max(0.0, x)
    x[x>0] = 1
    return x



