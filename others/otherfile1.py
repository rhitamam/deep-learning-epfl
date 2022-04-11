import torch
from torch.utils.data import random_split

def split_train_test (imgs, split_ratio = 0.8):
    return train, test = random_split(imgs, 
                                        [split_ratio * imgs.shape[0], (1 - split_ratio) * imgs.shape[0])