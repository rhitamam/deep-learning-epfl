from torch import empty , cat , arange, Tensor, manual_seed, repeat_interleave
from torch.nn.functional import fold, unfold
from Miniproject_2.others.otherfile1 import * #To uncomment for the final submission
#from others.otherfile1 import *
import random, math
import pickle
from pathlib import Path

random.seed(0)
manual_seed(0)

class Module(object):
    '''
    Abstract superclass that all modules must inherit in order to be sure that the forward and backward pass are implemented.
    '''
    def forward (self,  input) :
        raise NotImplementedError
    def backward (self,  gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []


class Conv2d(Module):
    '''
    Module to implement a 2D convolution over an input signal composed of several input planes.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        '''
        Constructor that instantiates all the parameters used in the convolution. 
        Initializes weights and bias such as the Conv2d counterpart.
        Inputs: 
            * in_channels (int) - Number of channels in the input image
            * out_channels (int) - Number of channels produced by the convolution
            * kernel_size (int or tuple) - Size of the convolving kernel
            * stride (int or tuple, optional) - Stride of the convolution. Default: 1
            * padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
            * dilation (int or tuple, optional) - Spacing between kernel elements. Default: 1
            * bias (bool, optional) - If True, adds a learnable bias to the output. Default: True
        '''
        self.in_chan = in_channels
        self.out_chan = out_channels
        self.kernel_size = int_to_tuple(kernel_size)
        self.stride = int_to_tuple(stride)
        self.padding = int_to_tuple(padding)
        self.dilation = int_to_tuple(dilation)
        self.bias_bool = bias
        self.weight, self.bias = initialize(in_channels, out_channels, self.kernel_size, bias)
        self.w_grad = empty(self.weight.shape)
        self.b_grad = empty(self.bias.shape)
        self.input = Tensor()
    

    def forward (self,  input) :
        """
        Return a convoluted tensor.
        Inputs:
            * input (tensor) - Tensor containing the input tensor to convolute with a tensor of weights
        Outputs: 
            * output (tensor) - Tensor resulting from the convolution
        """
        self.input = input.float()
        unfolded = unfold(input.float(), kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        wxb = self.weight.view(self.out_chan, -1) @ unfolded + self.bias.view(1, -1, 1)
        output = wxb.view(input.shape[0], 
                    self.out_chan,
                    math.floor(1+(input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0]),
                    math.floor(1+(input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[0]))
        return output

    def backward(self, gradwrtoutput):
        """
        Return gradient of the loss w.r.t. to the output of the previous layer.
        Compute the gradient of the loss w.r.t. the weights.
        Compute the gradient of the loss w.r.t. the weights the bias (if Conv2d was initialized with bias=True).
        Inputs:
            * gradwrtoutput (tensor) - Tensor containing the gradient of the loss w.r.t. the output of the layer
        Outputs: 
            * dl_dx_old (tensor) - Tensor containing the gradient of the loss w.r.t. the output of the previous layer
        """
        dl_dx_old = []
        a,b,c,d = self.input.shape
        
        unfolded = unfold(self.input, 
                            kernel_size=self.kernel_size, 
                            dilation=self.dilation, 
                            padding=self.padding, 
                            stride=self.stride)
        self.w_grad = ((gradwrtoutput.view(a,self.out_chan, -1) 
                        @ unfolded.transpose(1,2)).sum(0)).view(self.out_chan, self.in_chan, self.kernel_size[0], self.kernel_size[1])

        if self.bias_bool:
            self.b_grad = gradwrtoutput.sum((0,2,3))

        intermediate = gradwrtoutput.view(a, self.out_chan, -1).transpose(1,2) @ self.weight.view(self.out_chan, -1)

        dl_dx_old = fold(intermediate.transpose(1,2), 
                            (c,d),
                            kernel_size=self.kernel_size[0], 
                            dilation=self.dilation, 
                            padding=self.padding, 
                            stride=self.stride)

        return dl_dx_old 


    def param(self):
        """
        Return a list of pairs composed of a parameter tensor and a gradient tensor of the same size.
        Output: 
            * list of pairs composed of a parameter tensor and a gradient tensor
        """
        if self.bias_bool:
            return [(self.weight, self.w_grad), (self.bias, self.b_grad)]
        else:
            [(self.weight, self.w_grad)]


    def zero_grad(self):
        """
        Set all the gradient of the weight and the bias to zero.
        """
        self.w_grad = empty(self.weight.shape).normal_()
        if self.bias_bool:
            self.b_grad = empty(self.bias.shape).normal_()


    def set_weight_and_bias(self, weight, bias):
        """
        Weight and bias setter function.
        """
        self.weight = weight
        if self.bias_bool:
            self.bias = bias
        

class NearestUpSampling(Module) :
    '''
    Upsamples a given multi-channel data.
    '''
    def __init__(self,scalefactor):
        '''
        Instantiates the NearestUpSampling layer with its scale factor.
        Inputs:
            * scale_factor (float or Tuple[float, float], optional) - multiplier for spatial size.
        '''
        self.scalefactor = scalefactor
    

    def forward(self, input):
        '''
        Applies a 2D nearest neighbor upsampling to an input signal composed of several input channels.
        Inputs:
            * input (tensor) - Tensor containing the input tensor to upsample with a tensor of weights
        Outputs: 
            * output (tensor) - Tensor resulting from the upsampling
        '''
        first = repeat_interleave(input, 2, dim=3)
        final =  repeat_interleave(first,2,dim=2)
        return final


    def backward(self, gradwrtoutput):
        '''
        Return gradient of the loss w.r.t. to the output of the previous layer.
        Inputs:
            * gradwrtoutput (tensor) - Tensor containing the gradient of the loss w.r.t. the output of the layer
        Outputs: 
            * final (tensor) - Tensor containing the gradient of the loss w.r.t. the output of the previous layer
        '''
        if isinstance(gradwrtoutput,tuple):
            weight, bias = gradwrtoutput
        else:
            weight = gradwrtoutput
        u = unfold(weight,kernel_size=self.scalefactor, stride=2)
        viewed = u.view(weight.shape[0],
                        weight.size()[1],
                        self.scalefactor*self.scalefactor,
                        int((weight.size()[2]/self.scalefactor)*(weight.size()[2]/self.scalefactor)))
        to_mean = viewed.transpose(1,1).transpose(2,3)
        meaned = to_mean.sum(axis=3)
        final = meaned.view(weight.size()[0],
                            weight.size()[1],
                            int(weight.size()[2]/2),
                            int(weight.size()[2]/2))
        return final


class Sequential(Module):
    '''
    A sequential container of different modules.
    '''
    def __init__(self, *modules):
        """
        Constructor of the Sequential class, that is the concatenation of different modules.
        Inputs:
            * modules (Module): Ordered list of module forming the network.
        """
        self.modules = modules
        self.input = Tensor()


    def forward (self, input) :
        """
        Perform the forward pass calling one after the other the forward methods of the ordered modules contained in the list.
        Inputs:
            * input (tensor) - Tensor containing the input tensor 
        Outputs: 
            * output (tensor) - Tensor resulting from the sequential 
        """
        out = input
        for module in self.modules:
            out = module.forward(out)
        self.input = out
        return out
    

    def backward (self, gradwrtoutput):
        """
        Perform the backward pass calling in the reversed order the backward methods of the ordered modules contained in the list.
        Inputs:
            * gradwrtoutput (tensor) - Tensor containing the gradient of the loss w.r.t. the output of the last layer
        Outputs: 
            * final (tensor) - Tensor containing the gradient of the loss w.r.t. the output of the first layer
        """
        for module in reversed(self.modules):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput
    
    def param (self) :
        """
        Return parameters of the different layers of the network
        """
        param = [m.param() for m in self.modules]
        return param
    

    def zero_grad(self) :
        """
        Set all the gradient of the weight and the bias to zero
        """
        for m in self.modules:
            if isinstance(m, Conv2d):
                m.zero_grad()


class Model(Module):
    def __init__(self) :
        """
        Class constructor, instantiates model, optimizer, loss function, batch-norm, criterion and device.
        """
        #define the model
        self.model = Sequential(Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=2, padding=1), 
                ReLU(),
                Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1), 
                ReLU(), 
                NearestUpSampling(2), 
                Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), 
                ReLU(),
                NearestUpSampling(2), 
                Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                Sigmoid())

        #define the batch size
        self.mini_batch_size = 10

        #define the loss criterion
        self.criterion = MSE()

        #define the optimizer
        self.lr = 0.001
        self.momentum = 0.9
        self.optimizer = SGD(self.model, lr=self.lr, momentum=self.momentum)


    def set_lr(self, lr):
        '''
        SGD learning rate setter.
        Inputs:
            * lr (int) - learning rate
        '''
        self.lr = lr


    def set_momentum(self, momentum):
        '''
        SGD momentum setter.
        Inputs:
            * momentum (int) - momentum
        '''
        self.momentum = momentum
        
        
    def train(self, train_input, train_target, num_epochs, save_model = False) :
        """
        Train the model with train_input and train_target and save the trained model if argument save_model is seet to True.
        Inputs:
            * train_input (tensor) - Tensor containing a set of corrupted images (not normalized) to use for training the denoiser
            * train_target (tensor) - Tensor containing a set of corrupted images (not normalized) to use for computing the loss
            * num_epoch (int) - Number of the epochs to perform for the training
            * save_model (bool) - Default False.
        """
        #: normalize the input data
        train_input = train_input.float() / 255
        train_target= train_target.float() / 255

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
            self.save_model("Miniproject_2/bestmodel.pth")           

    def predict(self, test_input):
        """
        Predict the output of the model with test_input.
        Input:
            * test_input (tensor) - Tensor containing a set of corrupted images (not normalized) to use for training the denoiser
        Output:
            * Tensor containing not normalized denoised images predicted by the trained model.
        """
        return self.model.forward(test_input/255) * 255

    def load_pretrained_model(self):
        """
        Load the pretrained model saved as "bestmodel.pth" in model.
        """
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, "rb") as f:
            loaded_dict = pickle.load(f)
        i = 0
        for m in self.model.modules:
            if isinstance(m, Conv2d):
                weight = loaded_dict['c' + str(i)]['weight']
                bias = loaded_dict['c' + str(i)]["bias"]
                m.set_weight_and_bias(weight, bias)  
                i+=1 
        return self.model
        

    def save_model(self, FILE) :
        """
        Store each of the modules’ states in a pickle file

        Input:
            * FILE (string): Name of the file to save
        """
        print("save the model as " + FILE)
        
        #create the dictionnary
        model_dict = {}
        if isinstance(FILE, str):
            model_dict = {}
            i = 0
            for m in self.model.modules:
                if isinstance(m, Conv2d):
                    sub_dict= {}
                    sub_dict['weight'] = m.weight
                    sub_dict['bias'] = m.bias
                    model_dict['c' + str(i)] = sub_dict
                    i+=1
            
            with open(FILE, 'wb') as f:
                pickle.dump(model_dict, f)
        else:
            raise RuntimeError('Error: FILE must be a string')
            
    
    def psnr(self, denoised , ground_truth):
        """
        Computes the peak signal-to-noise ratio (PSNR) used to compare image compression quality.
        Inputs:
            * denoised (tensor) - denoised image
            * ground_truth (tensor) - target clean image
        Outputs: 
            * (int) - the PSNR error between the input and the target
        """
        # Peak Signal to Noise Ratio: denoised and ground_truth have range [0, 1] 
        denoised = denoised.float()
        ground_truth = ground_truth.float()
        mse = ((denoised - ground_truth) ** 2).mean()
        return -10 * math.log10(mse + 10**-8)
       

class SGD(object) :
    def __init__(self, module, lr=0.01, momentum=0.9):
        """
        Implements stochastic gradient descent (optionally with momentum).
        Inputs:
            * module (Module): Sequential model
            * lr (double): learning rate [0, 1]
            * momentum (double): [0,1]
        """
        self.module = module
        self.lr = lr
        self.momentum = momentum

    def step (self):
        """
        Update parameters of the network after one epoch
        """
        param = self.module.param()       
        for p, m in zip(param, self.module.modules):
            if isinstance(m, Conv2d):
                weight = p[0][0] - (self.lr * p[0][1])
                bias = p[1][0] - (self.lr * p[1][1])
                m.set_weight_and_bias(weight, bias)
    

    def zero_grad(self) :
        """
        Set all the gradient of the weight and the bias to zero.
        """
        self.module.zero_grad()


class MSE(Module) :
    '''
    Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input x and target y.
    '''
    def __init__(self):
        self.tensor = Tensor()
        self.target = Tensor()

    def forward (self, input, target):
        '''
        Computes MS error.
        Inputs:
            * input (tensor) - input to compare
            * target (tensor) - ground truth
        Outputs:
            * (int) - MSE error
        '''
        self.tensor = input
        self.target = target
        return MSE_func(input, target)

    def backward (self):
        '''
        Compute local gradient of MSE with respect to the input and target.
        Outputs:
            * () - derivative of MSE error
        '''
        return dMSE(self.tensor, self.target)/ (self.tensor.shape[0]*self.tensor.shape[1]*self.tensor.shape[2]*self.tensor.shape[3])

    def param (self) :
        return []


class ReLU(Module):
    '''
    Applies the rectified linear unit function element-wise.
    '''
    def __init__(self):
        self.tensor = Tensor()

    def forward (self,  input):
        '''
        Computes ReLU(x).
        Inputs:
            * input (tensor) - input tensor
        Outputs:
            * (tensor) - output tensor
        '''
        self.tensor = input
        return ReLU_func(input)

    def backward (self, gradwrtoutput):
        '''
        Compute gradient of ReLU(x).
        Inputs:
            * input (tensor) - input tensor
        Outputs:
            * (tensor) - output tensor
        '''
        return gradwrtoutput * dReLU(self.tensor)

    def param (self) :
        return []


class Sigmoid(Module):
    def __init__(self):
        self.tensor = Tensor()

    def forward (self,  input):
        '''
        Compute Sigmoid(x).
        Inputs:
            * input (tensor) - input tensor
        Outputs:
            * (tensor) - output tensor
        '''
        self.tensor = input
        return Sigmoid_func(input)

    def backward (self, gradwrtoutput):
        '''
        Compute gradient of ReLU(x).
        Inputs:
            * input (tensor) - input tensor
        Outputs:
            * (tensor) - output tensor
        '''
        return gradwrtoutput * dSigmoid(self.tensor)

    def param (self) :
        return []
