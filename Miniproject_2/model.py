from copyreg import pickle
from pickletools import optimize
from telnetlib import SE
from turtle import forward, update
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
        unfolded = unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        print(unfolded.shape)
        wxb = self.weights.view(self.out_chan, -1) @ unfolded + self.bias.view(1, -1, 1)
        output = wxb.view(input.shape[0], 
                    self.out_chan,
                    math.floor(1+(input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0]),
                    math.floor(1+(input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[0]))
        return output
        
    def backward (self,  gradwrtoutput):
        self.b_grad = gradwrtoutput.mean((0,2,3))
        unfolded = unfold(self.input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        self.w_grad =  gradwrtoutput @ unfolded.view((self.input.shape[0], self.input.shape[1], gradwrtoutput.shape[2], -1)) 
        return (self.w_grad, self.b_grad)

    def param (self):
        """
        Return a list of pairs composed of a parameter tensor and a gradient tensor of the same size
        Output: 
            * list of pairs composed of a parameter tensor and a gradient tensor
        """
        return [(self.weights, self.w_grad), (self.bias, self.b_grad)]

    def zero_grad(self):
        'set all the gradient of the weight and the bias to zero'
        self.w_grad = empty(self.weights.shape)
        self.b_grad = empty(self.bias.shape)

    def set_weights_and_bias(self, weights, bias):
        self.weights = weights
        self.bias = bias
        

class NearestUpsampling(Module) :
    def __init__(self,scalefactor):
        self.scalefactor = scalefactor
        
    
    def forward(self, input):
        first = repeat_interleave(input, 2, dim=3)
        final =  repeat_interleave(first,2,dim=2)
        return final

    def backward(self, gradwrtoutput):
         self.gradwrtoutput = gradwrtoutput
         #raise NotImplementedError


class Sequential(Module) :
    def __init__(self, *modules):
        self.modules = modules
        self.input = Tensor()

    def forward (self, input) :
        out = input
        for module in self.modules:
            out = module.forward(out)

        self.input = out
        return out
    
    def backward (self, gradwrtoutput):
        for module in reversed(self.modules):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput
    
    def param (self) :
        'should return multiple params'
        param = [m.param() for m in self.modules]
        return param
    
    def zero_grad(self) :
        'set all the gradient of the weight and the bias to zero'
        for m in self.modules:
            if isinstance(m, Conv2d):
                m.zero_grad()


class Model(Module):
    def __init__(self) :
        #define the model
        self.model = Sequential(Conv2d(in_channels= 3, out_channels= 4, stride= 1), ReLU(),
                                Conv2d(in_channels= 4, out_channels= 4, stride= 1), ReLU(), 
                                NearestUpsampling(2), 
                                Conv2d(8, 8, stride= 2), ReLU(),
                                NearestUpsampling(2),
                                Sigmoid())
        self.mini_batch_size = 100
        self.criterion = MSE()
        #define the optimizer
        self.lr = 0.1
        self.momentum = 0.9
        self.optimizer = SGD(self.model, lr= self.lr, momentum= self.momentum)

    def set_lr(self, lr):
        self.lr = lr

    def set_momentum(self, momentum):
        self.momentum = momentum
        
        
    def train(self, train_input, train_target, num_epochs, save_model = False) :
        for epoch in range(num_epochs) :
            acc_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                input = train_input[b: b + self.mini_batch_size]
                target = train_target[b: b+ self.mini_batch_size]
                #forward pass for the model sequential
                output = self.model.forward(input)
                #loss at the end of the forward pass
                loss = self.criterion.forward(output,target)
                acc_loss += loss

                #set the gradients to zero before starting to do backpropragation
                self.optimizer.zero_grad()
                #gradient of the loss wrt the output
                gradwrtout = self.criterion.backward()
                #backward pass for the model sequential
                self.model.backward(gradwrtout)

                #update according to the SGD optimizer
                self.optimizer.step()

            print(epoch, acc_loss)

        if save_model:
            self.save_model("bestmodel.pth")
            

    def predict(self, test_input):
        return self.model(test_input)

    def load_pretrained_model(self):
        with open("Miniproject_2/bestmodel.pth") as f:
            loaded_dict = pickle.load(f)
        for m in self.model.modules:
            weight = loaded_dict[str(m)]["weights"]
            bias = loaded_dict[str(m)]["bias"]
            m.set_weights_and_bias(weight, bias)   
        return self.model

    def save_model(self, FILE) :
        #store each of the modules’ states in a pickle file
        import pickle
        #make the dictionnary
        if isinstance(FILE, str):
            model_dict = {}
            for m in self.model.modules:
                if isinstance(m, Conv2d):
                    sub_dict= {}
                    sub_dict['weights'] = m.weights
                    sub_dict['bias'] = m.bias
                    model_dict[str(m)] = sub_dict
            
            with open(FILE, 'wb') as f:
                pickle.dump(model_dict, f)
        else:
            raise RuntimeError('Error: FILE must be a string')
       

class MSE(Module) :
    def __init__(self):
        self.tensor = Tensor()
        self.target = Tensor()

    def forward (self, input, target):
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
        

class SGD(object) :
    def __init__(self, module, lr=0.1, momentum=0.9):
        self.module = module
        self.lr = lr
        self.momentum = momentum

    def step (self):
        weight, w_grad, bias, b_grad = self.module.params()
        weight -= self.lr * w_grad
        bias -= self.lr * b_grad
        self.module.set_weight_and_bias(weight, bias)
    

    def zero_grad(self) :
        'set all the gradient of the weight and the bias to zero'
        self.module.zero_grad()

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


#==============================================================================
from charset_normalizer import from_path
import torch
from model import *
from torch.nn import functional
random.seed(0)
torch.manual_seed(0)

in_channels = 3 
out_channels = 4
kernel_size = (2, 3)

x = torch.randn((1, in_channels , 32, 32))
y = torch.randn((1, out_channels , 31, 30))

import torch.nn as nn

'''
model = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size),
          nn.ReLU(),
          nn.Sigmoid()
        )
'''
criterion = MSE()

seq = Sequential(Conv2d(in_channels, out_channels, kernel_size), ReLU(), Sigmoid())
'''seq = Sequential(Conv2d(in_channels= 3, out_channels= 4, kernel_size= kernel_size, stride= 1), ReLU(),
                                Conv2d(in_channels= 4, out_channels= 4,kernel_size= kernel_size, stride= 1), ReLU(), 
                                NearestUpsampling(2), 
                                Conv2d(8, 8, kernel_size= kernel_size, stride= 2), ReLU(),
                                NearestUpsampling(2),
                                Sigmoid())'''

optimizer= SGD(seq)
output = seq.forward(x)
loss = criterion.forward(output,y)
optimizer.zero_grad()
gradwrtout = criterion.backward()
seq.backward()

'''
#store each of the modules’ states in a pickle file
import pickle

FILE = 'bestmodel.pth'
#make the dictionnary
model_dict = {}
for m in seq.modules:
    if isinstance(m, Conv2d):
        sub_dict= {}
        sub_dict['weights'] = m.weights
        sub_dict['bias'] = m.bias
        #print(sub_dict)
        model_dict[str(m)] = sub_dict
        print(model_dict)
        
    with open(FILE, 'wb') as f:
        pickle.dump(model_dict, f)
'''
'''
with open("bestmodel.pth") as f:
    loaded_dict = pickle.load(f)
    for m in seq.modules:
        if isinstance(m, Conv2d):
            weight = loaded_dict[str(m)]["weights"]
            bias = loaded_dict[str(m)]["bias"]
            m.set_weights_and_bias(weight, bias)   
'''