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
from pro_cifar import pro_CIFAR100

import os
import argparse

from models import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from utils import FocalLoss
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model', '-m', type=int, help='resume from model file name')
    parser.add_argument('-p', default=1, type=float, help='proportion of old examples')
    parser.add_argument('--epoch', default=20, type=int, help='max epoch')
    parser.add_argument('--weighted', '-w', action='store_true', help='use a weighted loss for old examples')
    parser.add_argument('--fl', '-fl', action='store_true', help='use a focal loss')
    parser.add_argument('--hnm', '-hm', action='store_true', help='hard negative mining')    
    parser.add_argument('--ms', default=2, type=int, help='milestone point')    
    args = parser.parse_args()
    args.resume = True
    print(args)
    return args

mean = [x/255 for x in [125.3, 123.0, 113.9]]
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
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

final_train_acc = 1.
final_test_acc = 1.
perm = None
# Training
def sort_by_loss(args, net, epoch, optimizer, criterion, is_train=True):
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
    total_batch = len(trainloader)
    t = trange(total_batch)
    dataiter = iter(trainloader)
    train_loss = 0
    correct = 0
    if is_train:
        net.train()
    total = 0
    loss_list = np.zeros(50000, dtype=np.float32)
    idx = 0
    criterion = nn.CrossEntropyLoss(reduce=False)
    for batch_idx in t:
        inputs, targets = dataiter.next()
        t.set_description('Batch num %i %i' % (batch_idx, total_batch))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        if is_train:
            optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        #print(loss.shape[0])
        loss_list[idx:idx+loss.shape[0]] = loss
        idx = idx+loss.shape[0]
        if is_train:
            loss = loss.mean()
            loss.backward()
            optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        t.set_postfix(loss = '%.4f' % (train_loss/(batch_idx+1)), Acc = '%.4f' % (100.*correct/total))
    perm = np.argsort(loss_list)
    print(loss_list[49999], loss_list[5651])
    print(perm)
    reversed_perm = perm[::-1]
    return reversed_perm

def train(args, net, epoch, optimizer, criterion, first_epoch=False, use_all_data=False):
    global final_train_acc
    global perm
    NEW_LABEL_START = args.model
    PROPORTION = args.p
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if args.hnm and not use_all_data:
        is_train = not first_epoch
        print("is training?", is_train)
        perm = sort_by_loss(args, net, epoch, optimizer, criterion, is_train=is_train)

        #perm = np.random.permutation(50000)
    if args.hnm and not use_all_data:
        trainset = pro_CIFAR100(NEW_LABEL_START=NEW_LABEL_START, proportion=PROPORTION, root='./data',
                                train=True, download=True, transform=transform_train, perm=perm, sample_coefficient=2)
        print('using pro_CIFAR100')
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        print('using all data')
    if args.p == 0 and not use_all_data:
        return
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    total_batch = len(trainloader)
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
    final_train_acc = 100.*correct/total

def test(args, net, epoch, testloader, criterion):
    global best_acc
    global final_test_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    testiter = iter(testloader)
    total_test_batch = len(testloader)
    t = trange(total_test_batch)
    avg_outputs = np.zeros(100, dtype = np.float32)
    for batch_idx in t:
        inputs, targets = testiter.next()
        t.set_description('Batch num %i %i' % (batch_idx, total_test_batch))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs = net(inputs)
        # avg_outputs = avg_outputs*0.9 + outputs.cpu().data.numpy().sum(axis=0)*0.001
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        t.set_postfix(loss = '%.4f' % (test_loss/(batch_idx+1)), Acc = '%.4f' % (100.*correct/total))
    # print(avg_outputs)
    # print(avg_outputs.shape)
    test_with_category(args, net, testloader)
    # Save checkpoint.
    acc = 100.*correct/total
    final_test_acc = acc
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
    if epoch == args.epoch:
        #test_with_category(args, net, testloader)
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

def test_with_category(args, net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    #testiter = iter(testloader)
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
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
    acc = list(0. for i in range(100))
    for i in range(100):
        acc[i] = 100*class_correct[i]/class_total[i]
    print(acc)

def main():
    args = parse_args()
    MAX_EPOCH = args.epoch
    NEW_LABEL_START = args.model
    PROPORTION = args.p
    # Data
    print('==> Preparing data..')

    original_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

        checkpoint = torch.load('./checkpoint/'+'100_no'+str(args.model)+'.t7')
        net = checkpoint['net']
        best_acc = 0 #checkpoint['acc']
        start_epoch = 0 #checkpoint['epoch']
    else:
        print('==> Building model..')
        print('error!!!!!!')
        # net = VGG('VGG19')
        #net = ResNet20()
        #net = ResNet110()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    weights = [1 for i in range(100)]
    if args.weighted:
        for i in range(NEW_LABEL_START, 100):
            weights[i] = PROPORTION
    weights = torch.FloatTensor(weights)
    print(weights[0], weights[99])
    if use_cuda:
        weights = weights.cuda()

    criterion = nn.CrossEntropyLoss(weights)
    if args.fl:
        criterion = FocalLoss(100)
        print("####")
        criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    #MILESTONES = [int(MAX_EPOCH*0.5), int(MAX_EPOCH*0.75)]
    MILESTONES = [args.ms]
    scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)
    test_with_category(args, net, testloader)
    for epoch in range(start_epoch, MAX_EPOCH+1):
        use_all_data = False
        first_epoch = False
        if epoch == MAX_EPOCH:
            use_all_data = True
        # else:
        scheduler.step()
        if epoch == 0:
            first_epoch = True
        train(args=args, net=net, epoch=epoch, optimizer=optimizer, criterion=criterion, 
            first_epoch=first_epoch, use_all_data=use_all_data)
        test(args, net, epoch, testloader, criterion)

    print(final_train_acc, final_test_acc)

if __name__ == '__main__':
    main()