### For mini - project 1

def compute_output_size(W,K,P,S):
    return [(W−K+2P)/S]+1

def psnr(denoised , ground_truth):
    # Peak Signal to Noise Ratio: denoised and ground_truth have range [0, 1] 
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)
'''
list of questions for wednesday:
- how to translate the tf.concat into pytorch
- no feed-forward function? should we implement it into the __init__?
- DataLoaders or no need? if yes how to transform subset to dataloader?
'''

class Model(nn.Module) :
    def __init__ (self):
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        
        ksize = [1, 1, 2, 2]
        
        self.enc_conv1 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],48,0,3), 48, stride=3)
        self.enc_lr1 = nn.LeakyReLU(0.1)
        self.enc_pool1 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv2 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],48,0,3), 48, stride=3)
        self.enc_lr2 = nn.LeakyReLU(0.1)
        self.enc_pool2 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv3 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],48,0,3), 48, stride=3)
        self.enc_lr3 = nn.LeakyReLU(0.1)
        self.enc_pool3 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv4 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],48,0,3), 48, stride=3)
        self.enc_lr4 = nn.LeakyReLU(0.1)
        self.enc_pool4 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv5 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],48,0,3), 48, stride=3)
        self.enc_lr5 = nn.LeakyReLU(0.1)
        self.enc_pool5 = nn.MaxPool2d(ksize, stride=ksize, padding='same')
        
        self.enc_conv6 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],48,0,3), 48, stride=3)
        self.enc_lr6 = nn.LeakyReLU(0.1)
        
        #-----------------------------------------------
        
        self.upsample = nn.Upsample(self.shape[0])
        self.dec_conv5 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],96,0,3), 96, stride=3)
        self.dec_lr5 = nn.LeakyReLU(0.1)
        self.dec_conv5b = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],96,0,3), 96, stride=3)
        self.dec_lr5b = nn.LeakyReLU(0.1)
        
        self.upsample = nn.Upsample(self.shape[0])
        self.dec_conv4 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],96,0,3), 96, stride=3)
        self.dec_lr4 = nn.LeakyReLU(0.1)
        self.dec_conv4b = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],96,0,3), 96, stride=3)
        self.dec_lr4b = nn.LeakyReLU(0.1)
        
        self.upsample = nn.Upsample(self.shape[0])
        self.dec_conv3 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],96,0,3), 96, stride=3)
        self.dec_lr3 = nn.LeakyReLU(0.1)
        self.dec_conv3b = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],96,0,3), 96, stride=3)
        self.dec_lr3b = nn.LeakyReLU(0.1)
        
        self.upsample = nn.Upsample(self.shape[0])
        self.dec_conv2 = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],96,0,3), 96, stride=3)
        self.dec_lr2 = nn.LeakyReLU(0.1)
        self.dec_conv2b = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],96,0,3), 96, stride=3)
        self.dec_lr2b = nn.LeakyReLU(0.1)
        
        self.upsample = nn.Upsample(self.shape[0])
        self.dec_conv1a = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],64,0,3), 64, stride=3)
        self.dec_lr1a = nn.LeakyReLU(0.1)
        self.dec_conv1b = nn.Conv2d(self.shape[0], compute_output_size(self.shape[0],32,0,3), 32, stride=3)
        self.dec_lr1b = nn.LeakyReLU(0.1)
        
        self.dec_conv1 = nn.conv(self.shape[0], compute_output_size(self.shape[0],32,0,3), 3, stride=1)

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model
        pass

    def train(self, train_input, train_target):
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.
        optimizer = optim.Adam(model.parameters(), lr = 1e-1)
        nb_epochs = 250
        criterion = nn.MSELoss()
        eta = 1e-1
        for e in range(nb_epochs):
            acc_loss = 0

            for b in range(0, train_input.size(0), mini_batch_size):
                output = model(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                acc_loss = acc_loss + loss.item()

                model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for p in model.parameters():
                        p -= eta * p.grad

            # print(e, acc_loss)

    def predict(self, test_input):
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        pass