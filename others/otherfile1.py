import torch
from torch.utils.data import random_split
import torchvision.transforms as T

def data_transform(imgs):
    transform = T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    return transform(imgs)

def split_train_test (imgs, split_ratio = 0.8):
    return train, test = random_split(imgs, 
                                        [split_ratio * imgs.shape[0], (1 - split_ratio) * imgs.shape[0])