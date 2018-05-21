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
from resnext import * # import model file
from data import *
from utils import progress_bar # calculate time using
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
net = resnext101() # load model

# print(device)
if device == 'cuda':
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Data
TRAIN_LIST_PATH = './info/train_filelist_all.txt'
VALID_LIST_PATH = './info/val_filelist.txt'
BATCH_SZIE = 128 # Batch Size

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

trainset = webvisionData(TRAIN_LIST_PATH, 'train', transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SZIE, shuffle=True, num_workers=64)

testset = webvisionData(VALID_LIST_PATH, 'valid', transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SZIE, shuffle=True, num_workers=64)

# learning rate decay
def adjust_learning_rate(epoch):
    lr = args.lr
    if epoch > 4:
        lr = lr / 10.0
    if epoch > 7:
        lr = lr / 10.0
    if epoch > 9:
        lr = lr / 10.0
    if epoch > 10:
        lr = lr / 10.0

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
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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


for epoch in range(start_epoch, 12):
    adjust_learning_rate(epoch)
    train(epoch)
    test(epoch)
