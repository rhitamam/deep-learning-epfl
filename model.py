### For mini - project 1
from ast import Mod
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from others.otherfile1 import *
torch.manual_seed(0)


def psnr(denoised , ground_truth):
    # Peak Signal to Noise Ratio: denoised and ground_truth have range [0, 1] 
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)

class Model(nn.Module) :
    def __init__ (self):
        ## instantiate model + optimizer + loss function + any other stuff you need
        super(Model, self).__init__()
        self.autoencoder = nn.Sequential(
                                ## encoder
                                nn.Conv2d(3, 32, kernel_size = 3, stride = 1),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(32, 32, kernel_size = 3, stride = 1),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(32, 32, kernel_size = 3, stride = 1),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(32, 8, kernel_size = 3, stride = 1),
                                ## decoder
                                nn.ConvTranspose2d(8, 32, kernel_size=3, stride=1),
                                nn.LeakyReLU(0.1),
                                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
                                nn.LeakyReLU(0.1),
                                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
                                nn.LeakyReLU(0.1),
                                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
                                nn.LeakyReLU(0.1),
                                nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1)        
            )
        
        self.criterion = nn.MSELoss()
        self.nb_epochs = 250
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.mini_batch_size = 100
        self.eta = 1e-1
        

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model
        
        self.load_state_dict(torch.load("bestmodel.pth"))
        self.eval()
            



        pass

    def train(self, train_input, train_target):
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.
        for e in range(self.nb_epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.autoencoder(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                acc_loss = acc_loss + loss.item()
                
                self.autoencoder.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for p in self.autoencoder.parameters():
                        p -= self.eta * p.grad

            print(e, acc_loss)

    def predict(self, test_input):
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        return self.autoencoder(test_input)    


######################################################################

print('hello')

noisy_imgs_1 , noisy_imgs_2 = torch.load('train_data_.pkl')
noisy_imgs, clean_imgs = torch.load('val_data.pkl')
print('Shape of noisy_imgs_1', noisy_imgs_1.shape)
print('Shape of noisy_imgs_2', noisy_imgs_2.shape)
print('Shape of noisy_imgs', noisy_imgs.shape)
print('Shape of clean_imgs', clean_imgs.shape)

noisy_imgs_1 = noisy_imgs_1 / 255
noisy_imgs_2 = noisy_imgs_2 / 255
noisy_imgs = noisy_imgs / 255
clean_imgs = clean_imgs / 255

""" for k in range(10):
    model = Model()
    model.train(noisy_imgs_1, noisy_imgs_2)
    prediction = model.predict(noisy_imgs)
    nb_test_errors = psnr(prediction, clean_imgs)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0))) """

model = Model()

FILE = "bestmodel.pth"
torch.save(model.state_dict, FILE)


""" loaded = Model()
loaded.load_state_dict(torch.load(FILE))
loaded.eval() """


'''
TO DO :
- save and load_pretrained_model() for the best model 
- optimize our model
- test the predictions
'''