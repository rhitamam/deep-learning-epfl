#import torch
from torch import empty
import math


    
###### ReLU and its derivative ######

def ReLU(x):
    x_copy = x.clone()
    x_copy[x_copy<=0]=0
    return x_copy

def df_ReLU(x):
    x_copy = x.clone()
    x_copy[x_copy<=0] = 0
    x_copy[x_copy>0] = 1
    return x_copy



