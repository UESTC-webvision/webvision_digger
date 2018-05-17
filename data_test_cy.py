from __future__ import print_function, division
import os
import torch
# import pandas as pd
from torch.autograd import Variable
import random
from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

CLASS_NUM = 5000
VALID_PATH = '../val_images_resized/'


class webvisionData(Dataset):
    def __init__(self, data_path, kind, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.kind = kind
        self.fileListX, self.fileListY = [], []

        self.load_list()

    def __getitem__(self, index): # 获取一个index
        img_name = self.fileListX[index]
        label = self.fileListY[index]
        if self.kind == 'train':
            img_path = os.path.join('../', img_name)
        else:
            img_path = os.path.join(VALID_PATH, img_name)

        # with Image.open(img_path) as img:
        #    image = img.convert('RGB')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label#？转为one_HOT标签torch.Tensor([label]).uniform_(0, CLASS_NUM).long()

    def __len__(self): # 输出每一个epoch的迭代数
        return self.ListNum

    def load_list(self): # 加载这次需要处理的item
        file_list = open(self.data_path, 'r')

        # read filelist
        for fileItem in file_list:
            item = fileItem.split(' ')          #fileItem的格式：flickr_images_resized/q00071/6778321733.jpg 29
                                                #把label和文件名分割开
                                                #Item的格式['flickr_images_resized/q00071/26362919313.jpg', '29\n']

            self.fileListX.append(item[0])              #fileListX放文件名，格式['flickr_images_resized/q00071/26362919313.jpg']
            self.fileListY.append(int(item[1][:-1]))    #fileListY放label，只取到了数字，没有换行符，格式[29]
            # picAndLabel[int(item[1][:-1])].append(item[0])

        self.ListNum = len(self.fileListX)
        zip_data = list(zip(self.fileListX, self.fileListY)) #将原来的数组转换成了形如格式：('flickr_images_resized/q00046/10894955944.jpg', 21)
        #random.shuffle(zip_data)  #这里打乱这一步是什么意思？：先删除一下，不打乱的话是按照类别学习
        self.fileListX, self.fileListY = zip(*zip_data)