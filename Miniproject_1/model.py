### For mini - project 1
from ast import Mod
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#from others.otherfile1 import *
torch.manual_seed(0)


def psnr(denoised , ground_truth):
    # Peak Signal to Noise Ratio: denoised and ground_truth have range [0, 1] 
    denoised = denoised.float()/255
    ground_truth = ground_truth.float()/255
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)


#=====================================================================================

class myModel (nn.Module) :
    def __init__ (self):
        super(myModel, self).__init__()
        #define the convolution layer for the encoder
        #layer 1
        self.c0 = nn.Conv2d(3, 48, kernel_size = 3, stride = 1, padding="same")
        self.c1 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding="same")
        #layer 2
        self.c2 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding="same")
        #layer 3
        self.c3 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding="same")
        #layer 4
        self.c4 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding="same")
        #layer 5
        self.c5 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding="same")
        #layer 6
        self.c6 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding="same")      

        #define the activation function
        self.lr = nn.LeakyReLU(0.1)
        self.r = nn.ReLU()

        #define maxpooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #define normalization
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(3)

        #define upsampling for decoder part
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        #define the decoder
        self.ct5 = nn.ConvTranspose2d(96, 144, kernel_size=3, stride=1, padding=1)
        self.ct4 = nn.ConvTranspose2d(144, 96, kernel_size=3, stride=1, padding=1)
        self.ct3 = nn.ConvTranspose2d(144, 96, kernel_size=3, stride=1, padding=1)
        self.ct2 = nn.ConvTranspose2d(144, 96, kernel_size=3, stride=1, padding=1)
        self.ct1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.dec5a = nn.Conv2d(144, 96, kernel_size=3, stride=1, padding="same")
        self.dec5b = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding="same")

        self.dec4a = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding="same")
        self.dec4b = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding="same")

        self.dec3a = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding="same")
        self.dec3b = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding="same")

        self.dec2a = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding="same")
        self.dec2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same")

        self.dec1a = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same")
        self.dec1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same")

        self.out = nn.Conv2d(32, 3, kernel_size=1)
        self.linear = nn.Linear(3, 32)

    def forward(self, x):

        #encoder part
        #layer 1
        y = self.c0(x)
        y = self.bn1(y)
        y = self.lr(y)
        y = self.c1(y)
        y = self.bn1(y)
        y = self.lr(y)
        x1 = self.pool(y)
        #--------------------------
        #layer 2
        y = self.c2(x1)
        y = self.bn1(y)
        y = self.lr(y)
        x2 = self.pool(y)
        #--------------------------
        #layer 3
        y = self.c3(x2)
        y = self.bn1(y)
        y = self.lr(y)
        x3 = self.pool(y)
        #--------------------------
        #layer 4
        y = self.c4(x3)
        y = self.bn1(y)
        y = self.lr(y)
        x4 = self.pool(y)
        #--------------------------
        #layer 5
        y = self.c5(x4)
        y = self.bn1(y)
        y = self.lr(y)
        x5 = self.pool(y)
        #--------------------------
        #layer 6
        y = self.c6(x5)
        y = self.bn1(y)
        y = self.lr(y)

        #decoder part

        y = self.up(y)
        y = torch.cat([y, x4], dim = 1)
        y = self.ct5(y)
        y = self.dec5a(y)
        y = self.bn2(y)
        y = self.lr(y)
        y = self.dec5b(y)
        y = self.bn2(y)
        y = self.lr(y)
        #---------------------------------

        y = self.up(y)
        y = torch.cat([y, x3], dim = 1)
        y = self.ct4(y)
        y = self.dec4a(y)
        y = self.bn2(y)
        y = self.lr(y)
        y = self.dec4b(y)
        y = self.bn2(y)
        y = self.lr(y)
        #---------------------------------

        y = self.up(y)
        y = torch.cat([y, x2], dim = 1)
        y = self.ct3(y)
        y = self.dec3a(y)
        y = self.bn2(y)
        y = self.lr(y)
        y = self.dec3b(y)
        y = self.bn2(y)
        y = self.lr(y)
        #---------------------------------

        y = self.up(y)
        y = torch.cat([y, x1], dim = 1)
        y = self.ct2(y)
        y = self.dec2a(y)
        y = self.bn3(y)
        y = self.lr(y)
        y = self.dec2b(y)
        y = self.bn3(y)
        y = self.lr(y)
        #---------------------------------    

        y = self.up(y)
        y = self.ct1(y)
        y = self.dec1a(y)
        y = self.bn4(y)
        y = self.lr(y)
        y = self.dec1b(y)
        y = self.bn4(y)
        y = self.lr(y)
        #---------------------------------

        y = self.out(y)
        y = self.bn5(y)
        #y = self.linear(y)
        y = self.r(y)

        return y

#=====================================================================================
class Model(nn.Module) :
    def __init__ (self):
        ## instantiate model + optimizer + loss function + any other stuff you need
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = myModel().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.mini_batch_size = 100
        self.eta = 1e-1
        

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model
        return self.model.load_state_dict(torch.load("Miniproject_1/bestmodel.pth"))
        #self.model.eval()
            

    def train(self, train_input, train_target, num_epochs):
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.
        #: normalize the input data
        train_input = train_input.float() / 255
        train_target= train_target.float() / 255
        
        for e in range(num_epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.model(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                acc_loss = acc_loss + loss.item()
                
                self.model.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for p in self.model.parameters():
                        p -= self.eta * p.grad

            print(e, acc_loss)

        #: save the trained model
        FILE = "bestmodel.pth"
        torch.save(self.model.state_dict(), FILE)

    def predict(self, test_input):
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        return self.model(test_input.float()/255)        

#####################################################################
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noisy_imgs_1 , noisy_imgs_2 = torch.load('../data/train_data.pkl')
noisy_imgs_1 = noisy_imgs_1[:100].to(device)
noisy_imgs_2 = noisy_imgs_2[:100].to(device)
noisy_imgs, clean_imgs = torch.load('../data/val_data.pkl')
noisy_imgs = noisy_imgs.to(device)
clean_imgs = clean_imgs.to(device)

#noisy_imgs_1 = noisy_imgs_1.float() / 255
#noisy_imgs_2 = noisy_imgs_2.float() / 255
#noisy_imgs = noisy_imgs.float() / 255
#clean_imgs = clean_imgs.float() / 255

model = Model()
model.train(noisy_imgs_1, noisy_imgs_2, 10)
prediction = model.predict(noisy_imgs)
nb_test_errors = psnr(prediction, clean_imgs)
print('test error Net', nb_test_errors)
'''

''' loaded = Model()
loaded.load_state_dict(torch.load(FILE))
loaded.eval()'''
