'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from tqdm import trange

import os
import argparse

from models import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--model', '-m', default='best_ckpt', help='load model name')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
mean = [x/255 for x in [125.3, 123.0, 113.9]]
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, [1,1,1]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, [1,1,1]),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

total_batch = len(trainloader)
total_test_batch = len(testloader)
# Model

model_name = args.model + '.t7'
# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/'+model_name)
net = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

print('Best acc is %.3f' % best_acc)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    #testiter = iter(testloader)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    acc = list(0. for i in range(10))
    for i in range(10):
        acc[i] = 100*class_correct[i]/class_total[i]
    print(acc)
        #print('Accuracy of %5s : %2d %%' % (
        #    classes[i], 100 * class_correct[i] / class_total[i]))
    #t = trange(total_test_batch)
    #for batch_idx in t:
        #inputs, targets = testiter.next()
        #t.set_description('Batch num %i %i' % (batch_idx, total_batch))
        #if use_cuda:
            #inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        #outputs = net(inputs)
        #loss = criterion(outputs, targets)

        #test_loss += loss.data[0]
        #_, predicted = torch.max(outputs.data, 1)
        #total += targets.size(0)
        #correct += predicted.eq(targets.data).cpu().sum()

        #t.set_postfix(loss = '%.4f' % (test_loss/(batch_idx+1)), Acc = '%.4f' % (100.*correct/total))

test()
