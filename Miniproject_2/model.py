from torch import empty , cat , arange, Tensor, manual_seed, repeat_interleave
from torch.nn.functional import fold, unfold
#from Miniproject_2.others.otherfile1 import * #To uncomment for the final submission
from others.otherfile1 import *
import random, math
from torch import zeros

random.seed(0)
manual_seed(0)

class Module(object) :
    def forward (self,  input) :
        raise NotImplementedError
    def backward (self,  gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        self.in_chan = in_channels
        self.out_chan = out_channels
        self.kernel_size = int_to_tuple(kernel_size)
        self.stride = int_to_tuple(stride)
        self.padding = int_to_tuple(padding)
        self.dilation = int_to_tuple(dilation)
        self.bias_bool = bias
        self.weights, self.bias = initialize(in_channels, out_channels, self.kernel_size, bias)
        self.w_grad = empty(self.weights.shape)
        self.b_grad = empty(self.bias.shape)
        self.input = Tensor()
    

    def forward (self,  input) :
        self.input = input
        unfolded = unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        wxb = self.weights.view(self.out_chan, -1) @ unfolded + self.bias.view(1, -1, 1)
        if (torch.isnan(self.weights).any()):
                    raise RuntimeError('Error: Nan weights')
        output = wxb.view(input.shape[0], 
                    self.out_chan,
                    math.floor(1+(input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0]),
                    math.floor(1+(input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[0]))
        return output
        
    def backward_bis(self, gradwrtoutput):
        ''' 
        unfolded = unfold(self.input, kernel_size=self.kernel_size)
        self.w_grad = (gradwrtoutput.view(self.out_chan, -1) @ unfolded).view(self.w_grad.size())
        '''
        unfolded = unfold(self.input, 
                            kernel_size=self.kernel_size, 
                            dilation=self.dilation, 
                            padding=self.padding, 
                            stride=self.stride)
        #print ("unfolded", unfolded.view((-1, self.input.shape[1], gradwrtoutput.shape[2], gradwrtoutput.shape[3])).squeeze().shape)
        self.w_grad =  gradwrtoutput @ unfolded.view((self.input.shape[0], self.input.shape[1], gradwrtoutput.shape[2], -1))
        
        if self.bias_bool:
            self.b_grad = gradwrtoutput.sum((0,2,3))

        dl_dx_old = empty(self.input.shape)
        for idx, dl_ds in enumerate(gradwrtoutput):  # torch autograd jacobian cf slides torch autograd.grad
            dl_dx_prev = self.weights.view(self.out_chan, -1).t() @ dl_ds.view(1, self.out_chan, -1)
            # CHECK FIRST !!! FOLD UNFOLD CONSISTENCY
            dl_dx_prev_folded = fold(dl_dx_prev, output_size=self.input.size()[2:], kernel_size=self.kernel_size, stride=self.stride)
            dl_dx_old[idx] = dl_dx_prev_folded
        #dl_dx_old = dl_dx_old.view((self.input.shape[0], self.input.shape[1], self.input.shape[2],-1))
        '''
        dl_dx_old = fold(dl_dx_old, 
                        output_size=self.input.shape[2:], 
                        kernel_size=self.kernel_size, 
                        dilation=self.dilation, 
                        padding=self.padding, 
                        stride=self.stride)
                  '''  
        #print('w g', self.w_grad.shape)
        #print('b g', self.b_grad.shape)  
        #print("b conv out", dl_dx_old.shape)  
        print(dl_dx_old)

        return dl_dx_old
    

    def backward(self, gradwrtoutput):
        dl_dx_old = []
        #print("b conv")
        if (torch.isnan(gradwrtoutput).any()):
            raise RuntimeError('Error: Nan gwto')
        for dl_ds in gradwrtoutput:
            
            dl_dx_old_j = self.weights.view(self.out_chan, -1).t() @ dl_ds.view(1, self.out_chan, -1) 
            dl_dx_old_j = fold(dl_dx_old_j, output_size=self.input.shape[2:], kernel_size=self.kernel_size, dilation= self.dilation, padding= self.padding,stride=self.stride)
            dl_dx_old.append(dl_dx_old_j)
            
            for x_old in self.input:
                unfolded = unfold(x_old.unsqueeze(0), 
                                    kernel_size=self.kernel_size, 
                                    dilation=self.dilation, 
                                    padding=self.padding, 
                                    stride=self.stride)
                #print("unfold", unfolded.shape)
                if (torch.isnan(dl_ds).any()):
                    raise RuntimeError('Error: Nan dl_ds')
                
                self.w_grad.add_((dl_ds.reshape(self.out_chan, unfolded.shape[2]) @ unfolded.squeeze(0).t()).view(self.w_grad.size()))
                if (torch.isnan(self.w_grad).any()):
                    raise RuntimeError('Error: Nan w grad add')
        
        if self.bias_bool:
            self.b_grad = gradwrtoutput.sum((0,2,3))
        
        dl_dx_old = torch.cat(dl_dx_old)
        return (dl_dx_old)


    def param(self):
        """
        Return a list of pairs composed of a parameter tensor and a gradient tensor of the same size
        Output: 
            * list of pairs composed of a parameter tensor and a gradient tensor
        """
        if self.bias_bool:
            return [(self.weights, self.w_grad), (self.bias, self.b_grad)]
        else:
            [(self.weights, self.w_grad)]


    def zero_grad(self):
        'set all the gradient of the weight and the bias to zero'
        self.w_grad = zeros(self.weights.shape)
        if self.bias_bool:
            self.b_grad = zeros(self.bias.shape)


    def set_weights_and_bias(self, weights, bias):
        self.weights = weights
        if self.bias_bool:
            self.bias = bias
        

class NearestUpSampling(Module) :
    def __init__(self,scalefactor):
        self.scalefactor = scalefactor
    

    def forward(self, input):
        first = repeat_interleave(input, 2, dim=3)
        final =  repeat_interleave(first,2,dim=2)
        return final


    def backward(self, gradwrtoutput):
        if isinstance(gradwrtoutput,tuple):
            weights, bias = gradwrtoutput
        else:
            weights = gradwrtoutput
        u = unfold(weights,kernel_size=self.scalefactor, stride=2)
        viewed = u.view(weights.shape[0],
                        weights.size()[1],
                        self.scalefactor*self.scalefactor,
                        int((weights.size()[2]/self.scalefactor)*(weights.size()[2]/self.scalefactor)))
        to_mean = viewed.transpose(1,1).transpose(2,3)
        meaned = to_mean.mean(axis=3)
        final = meaned.view(weights.size()[0],
                            weights.size()[1],
                            int(weights.size()[2]/2),
                            int(weights.size()[2]/2))
        return final


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
        in_channels = 3
        out_channels = 3
        self.model = Sequential(Conv2d(in_channels=in_channels, out_channels=48, kernel_size=(3,3), stride=(2,2), padding=(1,1)), 
                ReLU(),
                Conv2d(in_channels= 48, out_channels= 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)), 
                ReLU(), 
                NearestUpSampling(2), 
                Conv2d(in_channels=64, out_channels= 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)), 
                ReLU(),
                NearestUpSampling(2), 
                Conv2d(in_channels=32, out_channels= 3, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                Sigmoid())

        self.mini_batch_size = 100
        self.criterion = MSE()
        #define the optimizer
        self.lr = 0.1
        self.momentum = 0.9
        self.optimizer = SGD(self.model, lr=self.lr, momentum=self.momentum)


    def set_lr(self, lr):
        self.lr = lr


    def set_momentum(self, momentum):
        self.momentum = momentum
        
        
    def train(self, train_input, train_target, num_epochs, save_model = False) :

        #: normalize the input data
        train_input = train_input.float() / 255
        train_target= train_target.float() / 255

        for epoch in range(num_epochs) :
            acc_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                
                input = train_input[b: b + self.mini_batch_size]
                if (torch.isnan(input).any()):
                    raise RuntimeError('Error: Nan inpout')
                target = train_target[b: b+ self.mini_batch_size]

                #forward pass for the model sequential  
                output = self.model.forward(input)
                print(output)

                #loss at the end of the forward pass
                loss = self.criterion.forward(output,target)

                #print('loss', loss)
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
        return self.model.forward(test_input)

    def load_pretrained_model(self):
        import pickle
        with open("Miniproject_2/bestmodel.pth", "rb") as f:
            loaded_dict = pickle.load(f)
        i = 0
        for m in self.model.modules:
            if isinstance(m, Conv2d):
                weight = loaded_dict['c' + str(i)]["weights"]
                bias = loaded_dict['c' + str(i)]["bias"]
                m.set_weights_and_bias(weight, bias)   
        return self.model

    def save_model(self, FILE) :
        #store each of the modules’ states in a pickle file
        import pickle
        #make the dictionnary
        if isinstance(FILE, str):
            model_dict = {}
            i = 0
            for m in self.model.modules:
                if isinstance(m, Conv2d):
                    sub_dict= {}
                    sub_dict['weights'] = m.weights
                    sub_dict['bias'] = m.bias
                    model_dict['c' + str(i)] = sub_dict
                    i+=1
            
            with open(FILE, 'wb') as f:
                pickle.dump(model_dict, f)
        else:
            raise RuntimeError('Error: FILE must be a string')
    
    def psnr(self, denoised , ground_truth):
        # Peak Signal to Noise Ratio: denoised and ground_truth have range [0, 1] 
        denoised = denoised.float()
        ground_truth = ground_truth.float()/255
        mse = torch.mean((denoised - ground_truth) ** 2)
        return -10 * torch.log10(mse + 10**-8)
       

class SGD(object) :
    def __init__(self, module, lr=0.1, momentum=0.9):
        self.module = module
        self.lr = lr
        self.momentum = momentum

    def step (self):
        param = self.module.param()       
        for p, m in zip(param, self.module.modules):
            if isinstance(m, Conv2d):
                weights = p[0][0] - (self.lr * p[0][1])
                bias = p[1][0] - (self.lr * p[1][1])
                m.set_weights_and_bias(weights, bias)
    

    def zero_grad(self) :
        'set all the gradient of the weight and the bias to zero'
        self.module.zero_grad()


class MSE(Module) :
    def __init__(self):
        self.tensor = Tensor()
        self.target = Tensor()

    def forward (self, input, target):
        self.tensor = input
        self.target = target
        #print('mse', MSE_func(input, target))
        return MSE_func(input, target)/ self.tensor.shape[0]

    def backward (self):
        #print("dmse", (dMSE(self.tensor, self.target))/ self.tensor.shape[0])
        return dMSE(self.tensor, self.target)/ self.tensor.shape[0]

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


class Sigmoid(Module):
    def __init__(self):
        self.tensor = Tensor()

    def forward (self,  input):
        self.tensor = input
        return Sigmoid_func(input)

    def backward (self, gradwrtoutput):
        return gradwrtoutput * dSigmoid(self.tensor)

    def param (self) :
        return []
        

#==============================================================================

import torch
from torch.nn import functional
random.seed(0)
torch.manual_seed(0)

noisy_imgs_1 , noisy_imgs_2 = torch.load('../data/train_data.pkl')
noisy_imgs_1 = noisy_imgs_1[:10].float()
noisy_imgs_2 = noisy_imgs_2[:10].float()
noisy_imgs, clean_imgs = torch.load('../data/val_data.pkl')
noisy_imgs = noisy_imgs[:10].float()
clean_imgs = clean_imgs[:10].float() / 255.0

model = Model()
model.train(noisy_imgs_1, noisy_imgs_2, 4)
prediction = model.predict(noisy_imgs)
prediction = prediction / 255.0
nb_test_errors = model.psnr(prediction, clean_imgs)
print('test error Net', nb_test_errors)

