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
from custom_cifar_100 import custom_CIFAR100
from custom_cifar import custom_CIFAR10
from proportion_cifar_100 import pro_CIFAR100

import os
import argparse

from models import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', '-m', help='resume from model file name')
parser.add_argument('--classnum', '-c', type=int, default=100, help='cifar class number')
parser.add_argument('--proportion', '-p', type=float, default=1.0, help='proportion')
args = parser.parse_args()

MAX_EPOCH = 160
MILESTONES = [int(MAX_EPOCH*0.5), int(MAX_EPOCH*0.75)]

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
mean = [x/255 for x in [125.3, 123.0, 113.9]]
acc_history = [0 for i in range(101)]

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

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.model+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    MAX_EPOCH = MAX_EPOCH - start_epoch
    start_epoch = 0
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    #net = ResNet110_100()
    #net = ResNet20_100()
    if args.classnum == 100:
        net = MobileNetV2(100)
        #net = ResNet32_100()
        #net = ShuffleNet(num_classes=100, in_channels=3)
    else:
        net = MobileNetV2()
        #net = ResNet32()
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

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch, trainloader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    total_batch = len(trainloader)
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

def test(epoch, testloader, MAX_EPOCH):
    global best_acc
    global acc_history
    total_test_batch = len(testloader)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    testiter = iter(testloader)
    t = trange(total_test_batch)
    for batch_idx in t:
        inputs, targets = testiter.next()
        t.set_description('Batch num %i %i' % (batch_idx, total_test_batch))
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
        #print('Saving..')
        #state = {
            #'net': net.module if use_cuda else net,
            #'acc': acc,
            #'epoch': epoch,
        #}
        #if not os.path.isdir('checkpoint'):
            #os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/best_ckpt.t7')
        best_acc = acc
    if epoch == MAX_EPOCH-1:
        #print('Saving..')
        #state = {
            #'net': net.module if use_cuda else net,
            #'acc': acc,
            #'epoch': epoch,
        #}
        #if not os.path.isdir('checkpoint'):
            #os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/last_ckpt%.3f.t7'%acc)
        test_with_category()
        #print('Best accuracy is %.3f' % best_acc)
    return acc

def test_with_category():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    C = args.classnum
    #testiter = iter(testloader)
    class_correct = list(0. for i in range(C))
    class_total = list(0. for i in range(C))
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
    acc = list(0. for i in range(C))
    for i in range(C):
        if class_total[i] == 0:
            acc[i] = 0
        else:
            acc[i] = 100*class_correct[i]/class_total[i]
    print(acc)


start_point = 10
if args.classnum == 10:
    start_point = 2
for cur_classnum in range(start_point, args.classnum+1):
    print('curently adding class number:', cur_classnum)
    if args.proportion > 0.99:
        if args.classnum == 100:
            trainset = custom_CIFAR100(root='./data', train=True, download=True, transform=transform_train, erase_start=cur_classnum)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=5, pin_memory=True)
            testset = custom_CIFAR100(root='./data', train=False, download=True, transform=transform_test, erase_start=cur_classnum)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=5, pin_memory=True)
        else:
            trainset = custom_CIFAR10(root='./data', train=True, download=True, transform=transform_train, erase_start=cur_classnum)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=5, pin_memory=True)
            testset = custom_CIFAR10(root='./data', train=False, download=True, transform=transform_test, erase_start=cur_classnum)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=5, pin_memory=True)
    else:
        trainset = custom_CIFAR10(root='./data', train=True, download=True, transform=transform_train, erase_start=cur_classnum)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=5, pin_memory=True)
        testset = custom_CIFAR10(root='./data', train=False, download=True, transform=transform_test, erase_start=cur_classnum)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=5, pin_memory=True)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    max_epoch = 2
    if cur_classnum == start_point:
        max_epoch = 50
    scheduler = MultiStepLR(optimizer, milestones=[max_epoch/2], gamma=0.1)
    acc = 0
    for epoch in range(0,max_epoch):
        scheduler.step()
        train(epoch, trainloader, optimizer)
        acc = test(epoch, testloader, MAX_EPOCH=max_epoch)
    acc_history[cur_classnum] = acc
print(acc_history)
