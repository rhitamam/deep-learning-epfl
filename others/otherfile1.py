import torch
from torch.utils.data import random_split

def split_train_test (imgs, split_ratio = 0.8):
    #train, test = torch.utils.data.random_split(imgs, [40000, 10000])
    train, test = torch.utils.data.random_split(imgs, [int(split_ratio * imgs.shape[0]), imgs.shape[0] - int(split_ratio * imgs.shape[0])])
    return train.dataset, test.dataset