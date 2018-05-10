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

CLASS_NUM = 5000
VALID_PATH = './val_images_resized/'


class webvisionData(Dataset):
    def __init__(self, data_path, kind, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.kind = kind
        self.fileListX, self.fileListY = [], []

        self.load_list()

    def __getitem__(self, index): # 获取一个index
        X, Y = self.read_data(index)
        return X, Y

    def __len__(self): # 输出每一个epoch的迭代数
        return self.ListNum

    def load_list(self): # 加载这次需要处理的item
        file_list = open(self.data_path, 'r')

        # read filelist
        for fileItem in file_list:
            item = fileItem.split(' ')
            self.fileListX.append(item[0])
            self.fileListY.append(int(item[1][:-1]))
            # picAndLabel[int(item[1][:-1])].append(item[0])

        self.ListNum = len(self.fileListX)
        zip_data = list(zip(self.fileListX, self.fileListY))
        random.shuffle(zip_data)  #random is not defined ,import random   --CY
        self.fileListX, self.fileListY = zip(*zip_data)

    def read_data(self, index):
        img_name = self.fileListX[index]
        if self.kind == 'train':
            img_path = os.path.join('./', img_name)
        else:
            img_path = os.path.join(VALID_PATH, img_name)

        image = io.imread(img_name)
        image = image.convert('RGB')

        if self.kind == 'train':
            image.resize((256, 256), Image.ANTIALIAS)
        else:
            image.resize((224, 224), Image.ANTIALIAS)
        image = self.transform(image)

        return image, torch.Tensor(label)
