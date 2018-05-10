import torchvision.datasets as datasets
import numpy as np
dataloader = datasets.CIFAR10
num_classes = 10
trainset = dataloader(root='./', train=True, download=True, transform=None)
print (trainset[0].shape)
print (trainset[1].shape)