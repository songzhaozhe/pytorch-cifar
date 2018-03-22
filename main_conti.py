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
from custom_cifar import custom_CIFAR10

import os
import argparse

from models import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', '-m', help='resume from model file name')
args = parser.parse_args()
args.resume = True

MAX_EPOCH = 30

MILESTONES = [int(MAX_EPOCH*0.5)]

mean = [x/255 for x in [125.3, 123.0, 113.9]]

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


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
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.model+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    start_epoch = 0
else:
    print('==> Building model..')
    print('error!!!!!!')
    # net = VGG('VGG19')
    net = ResNet20()
    #net = ResNet110()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t = trange(total_batch)
    dataiter = iter(trainloader)

    for batch_idx in t:
        inputs, targets = dataiter.next()
        t.set_description('Batch num %i %i' % (batch_idx, total_batch))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        t.set_postfix(loss = '%.4f' % (train_loss/(batch_idx+1)), Acc = '%.4f' % (100.*correct/total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    testiter = iter(testloader)
    t = trange(total_test_batch)
    for batch_idx in t:
        inputs, targets = testiter.next()
        t.set_description('Batch num %i %i' % (batch_idx, total_batch))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        t.set_postfix(loss = '%.4f' % (test_loss/(batch_idx+1)), Acc = '%.4f' % (100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/best_ckpt.t7')
        best_acc = acc
    if epoch == MAX_EPOCH:
        test_with_category()
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/last_ckpt%.3f.t7'%acc)
        print('Best accuracy is %.3f' % best_acc)

def test_with_category():
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

for epoch in range(start_epoch, MAX_EPOCH+1):
    scheduler.step()
    train(epoch)
    test(epoch)
