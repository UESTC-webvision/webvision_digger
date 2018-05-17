'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from resnet import *
from data import *
#from utils import progress_bar # calculate time using
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')
net = resnet50()
print(device)
if device == 'cuda':
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))  #实现模块级别上的数据并行
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']                     #一些参数传递的方式

criterion = nn.CrossEntropyLoss()                         #多分类任务的评估函数：交叉熵
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Data
TRAIN_LIST_PATH = '../info/train_filelist_all.txt'
VALID_LIST_PATH = '../info/val_filelist.txt'
BATCH_SZIE = 640

print('==> Preparing data..')
transform_train = transforms.Compose([               #数据预处理部分
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([               #测试集合预处理
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #
])

trainset = webvisionData(TRAIN_LIST_PATH, 'train', transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SZIE, shuffle=True, num_workers=32)   #生成一个迭代器

testset = webvisionData(VALID_LIST_PATH, 'valid', transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SZIE, shuffle=True, num_workers=32)
# learning rate decay
def adjust_learning_rate(epoch):
    lr = args.lr * (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.view(-1)   #将targets展成一维,我把这个给提前了，可能是one-hot标签没有处理好
#        print("print targets.size1: %d \n print targets: %d", targets.size(),targets)
        inputs, targets = inputs.cuda(), targets.cuda()   #转换为GPU处理的张量
#        print("print inputs.size:",inputs.size())
#        print("print targets.size2:",targets.size())
        optimizer.zero_grad()        #梯度清零
        inputs, targets = Variable(inputs), Variable(targets)#放进用于计算的Variable中
        outputs = net(inputs)
        # print(outputs)
        # print(targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]          #item改了，原本是dataloss.item()
        print("it's loss.item: %d ", loss.data[0])
        _, predicted = torch.max(outputs.data, 1)    #改了，原本是outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()#改了，原本是sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.view(-1)            #提前了
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]              #有改动，原本是loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+10):
    adjust_learning_rate(epoch)
    train(epoch)
    test(epoch)
