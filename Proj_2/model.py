from torch import empty , cat , arange
from torch.nn.functional import fold, unfold
torch.set ̇grad ̇enabled(False)


class Module (object) :
    def forward (self, *input) :
        raise NotImplementedError
    def backward (self, *gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []

'''
• Convolution layer.
• Transpose convolution layer, or alternatively a combination of Nearest neighbor upsampling + Convolution.
• Upsampling layer, which is usually implemented with transposed convolution, but you can alternatively use a combination of Nearest neighbor upsampling + Convolution for this mini-project.
• ReLU
• Sigmoid
• A container like torch.nn.Sequential to put together an arbitrary configuration of modules together.
• Mean Squared Error as a Loss Function
• Stochastic Gradient Descent (SGD) optimizer
'''