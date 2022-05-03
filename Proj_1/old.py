### For mini - project 1
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from others.otherfile1 import *
torch.manual_seed(0)

def compute_output_size(W,K,P,S):
    print(((W-K+2*P)/S)+1)
    return ((W-K+2*P)/S)+1

def psnr(denoised , ground_truth):
    # Peak Signal to Noise Ratio: denoised and ground_truth have range [0, 1] 
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)
'''
list of questions for wednesday:
- how to translate the tf.concat into pytorch
- no feed-forward function? should we implement it into the __init__?
- DataLoaders or no need? if yes how to transform subset to dataloader?
- is it conv2d or functional?
'''


class Model(nn.Module) :
    def __init__ (self):
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        
        self.enc_conv1 = nn.Conv2d(3, 48, kernel_size=(5,5), stride=3)
        self.enc_lr1 = nn.LeakyReLU(0.1)
        self.enc_pool1 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv2 = nn.Conv2d(3, 48, kernel_size=(5,5), stride=3)
        self.enc_lr2 = nn.LeakyReLU(0.1)
        self.enc_pool2 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv3 = nn.Conv2d(3, 48, kernel_size=(5,5), stride=3)
        self.enc_lr3 = nn.LeakyReLU(0.1)
        self.enc_pool3 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv4 = nn.Conv2d(3, 48, kernel_size=(5,5), stride=3)
        self.enc_lr4 = nn.LeakyReLU(0.1)
        self.enc_pool4 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv5 = nn.Conv2d(3, 48, kernel_size=(5,5), stride=3)
        self.enc_lr5 = nn.LeakyReLU(0.1)
        self.enc_pool5 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv6 = nn.Conv2d(3, 48, kernel_size=(5,5), stride=3)
        self.enc_lr6 = nn.LeakyReLU(0.1)
        
        #-----------------------------------------------
        
        self.upsample5 = nn.Upsample(3)
        self.dec_conv5 = nn.Conv2d(3, 96, kernel_size=(5,5), stride=3)
        self.dec_lr5 = nn.LeakyReLU(0.1)
        self.dec_conv5b = nn.Conv2d(3, 96, kernel_size=(5,5), stride=3)
        self.dec_lr5b = nn.LeakyReLU(0.1)
        
        self.upsample4 = nn.Upsample(3)
        self.dec_conv4 = nn.Conv2d(3, 96, kernel_size=(5,5), stride=3)
        self.dec_lr4 = nn.LeakyReLU(0.1)
        self.dec_conv4b = nn.Conv2d(3, 96, kernel_size=(5,5), stride=3)
        self.dec_lr4b = nn.LeakyReLU(0.1)
        
        self.upsample3 = nn.Upsample(3)
        self.dec_conv3 = nn.Conv2d(3, 96, kernel_size=(5,5), stride=3)
        self.dec_lr3 = nn.LeakyReLU(0.1)
        self.dec_conv3b = nn.Conv2d(3, 96, kernel_size=(5,5), stride=3)
        self.dec_lr3b = nn.LeakyReLU(0.1)
        
        self.upsample2 = nn.Upsample(3)
        self.dec_conv2 = nn.Conv2d(3, 96, kernel_size=(5,5), stride=3)
        self.dec_lr2 = nn.LeakyReLU(0.1)
        self.dec_conv2b = nn.Conv2d(3, 96, kernel_size=(5,5), stride=3)
        self.dec_lr2b = nn.LeakyReLU(0.1)
        
        self.upsample1 = nn.Upsample(3)
        self.dec_conv1a = nn.Conv2d(3, 64, kernel_size=(5,5), stride=3)
        self.dec_lr1a = nn.LeakyReLU(0.1)
        self.dec_conv1b = nn.Conv2d(3, 64, kernel_size=(5,5), stride=3)
        self.dec_lr1b = nn.LeakyReLU(0.1)
        
        self.dec_conv1 = nn.Conv1d(3, 3, kernel_size=(5,5), stride=1)

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model
        pass

    def train(self, train_input, train_target):
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
        nb_epochs = 250
        criterion = nn.MSELoss()
        eta = 1e-1
        mini_batch_size = 100
        for e in range(nb_epochs):
            acc_loss = 0

            for b in range(0, train_input.size(0), mini_batch_size):
                print(train_input.shape)
                output = model(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                acc_loss = acc_loss + loss.item()

                model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for p in model.parameters():
                        p -= eta * p.grad

            print(e, acc_loss)

    def predict(self, test_input):
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        pass
    
    
    def forward(self, x):
        x = self.enc_conv1(x)
        x = self.enc_lr1(x)
        x = self.enc_pool1(x)
        x = self.enc_conv2(x)
        x = self.enc_lr2(x)
        x = self.enc_pool2(x)
        x = self.enc_conv3(x)
        x = self.enc_lr3(x)
        x = self.enc_pool3(x)
        x = self.enc_conv4(x)
        x = self.enc_lr4(x)
        x = self.enc_pool4(x)
        x = self.enc_conv5(x)
        x = self.enc_lr5(x)
        x = self.enc_pool5(x)
        x = self.enc_conv6(x)
        x = self.enc_lr6(x)

        x = self.upsample5(x)
        x = self.dec_conv5(x)
        x = self.dec_lr5(x)
        x = self.dec_conv5b(x)
        x = self.dec_lr5b(x)
        x = self.upsample4(x)
        x = self.dec_conv4(x)
        x = self.dec_lr4(x)
        x = self.dec_conv4b(x)
        x = self.dec_lr4b(x)
        x = self.upsample3(x)
        x = self.dec_conv3(x)
        x = self.dec_lr3(x)
        x = self.dec_conv3b(x)
        x = self.dec_lr3b(x)
        x = self.upsample2(x)
        x = self.dec_conv2(x)
        x = self.dec_lr2(x)
        x = self.dec_conv2b(x)
        x = self.dec_lr2b(x)
        x = self.upsample1(x)
        x = self.dec_conv1(x)
        x = self.dec_lr1(x)
        x = self.dec_conv1b(x)
        x = self.dec_lr1b(x)
        
        x = self.dec_conv1(x)
        return x

    

    


######################################################################

print('hello')

noisy_imgs_1 , noisy_imgs_2 = torch.load('data/train_data.pkl')
noisy_imgs, clean_imgs = torch.load('data/val_data.pkl')
print('Shape of noisy_imgs_1', noisy_imgs_1.shape)
print('Shape of noisy_imgs_2', noisy_imgs_2.shape)
print('Shape of noisy_imgs', noisy_imgs.shape)
print('Shape of clean_imgs', clean_imgs.shape)
train_set_1, test_set_1 = torch.utils.data.random_split(noisy_imgs_1, [40000, 10000])
train_set_1 = train_set_1.dataset
test_set_1 = test_set_1.dataset
train_set_2, test_set_2 = torch.utils.data.random_split(noisy_imgs_2, [40000, 10000])
train_set_2 = train_set_2.dataset
test_set_2 = test_set_2.dataset
print('Shape of train_set_1', train_set_1.shape)
print('Shape of test_set_1', test_set_1.shape)
print('Shape of train_set_2', train_set_2.shape)
print('Shape of test_set_2', test_set_2.shape)

for k in range(10):
    model = Model()
    model.train(train_set_1, test_set_1)
    nb_test_errors = psnr(train_set_2, test_set_2)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))