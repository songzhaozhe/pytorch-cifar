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
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from utils import init_params

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', '-m', help='resume from model file name')
args = parser.parse_args()

max_epoch = 164

def schedule(epoch):
    #if (epoch < 500):
    #    return 0.01
    if (epoch <= 0.5*max_epoch):
        return 0.1
    elif (epoch <= 0.75*max_epoch):
        return 0.01
    else:
        return 0.001


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


mean = [x/255 for x in [125.3, 123.0, 113.9]]

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, [1,1,1]),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, [1,1,1]
        ),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=5, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=5, pin_memory=True)

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
    max_epoch = max_epoch - start_epoch
    start_epoch = 0
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    #net = ResNet32()
    net = ResNet20()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    #init_params(net)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
#scheduler = LambdaLR(optimizer, schedule)
scheduler = MultiStepLR(optimizer, milestones=[int(max_epoch*0.5), int(max_epoch*0.75)], gamma=0.1)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    acc = 0
    t = trange(total_batch)
    dataiter = iter(trainloader)

    for batch_idx in t:
        inputs, targets = dataiter.next()
        t.set_description('Batch num %i %i' % (batch_idx, total_batch))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = train_loss*0.2 + loss.data[0] * 0.8
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()
        acc = acc * 0.2 + 0.8 * 100. * correct / total
        t.set_postfix(loss = '%.4f' % train_loss, Acc = '%.4f' % acc)

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
    if epoch == max_epoch:
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

def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = schedule(epoch)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

for epoch in range(start_epoch, max_epoch+1):
    #adjust_learning_rate(optimizer, epoch)
    scheduler.step()
    train(epoch)
    test(epoch)
