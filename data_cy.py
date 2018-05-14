from __future__ import print_function, division
import os
import torch
# import pandas as pd
import random
from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class Webvisiondata(Dataset):
    def __init__(self, txt, transform=None,loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))               #用img这个list来包含图片的地址和标签，words[0],words[1]
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):                                #getitem_获取图片的索引，fn代表地址，label为标签
        fn, label = self.imgs[index]
        img = self.loader(fn)                                    #加载这张图片，img是图片
        if self.transform is not None:
            img = self.transform(img)                            #对图像进行简单的预处理
        return img, label

    def __len__(self):
        return len(self.imgs)                                   #返回数据集的长度

