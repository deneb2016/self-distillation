import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import numpy as np
import math
from model.resnet import ResNet
from model.densenet import DenseNet3
from model.vggnet import VGGNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR')

parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--net', type=str, help='model')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--model_path', type=str)
# set training session

parser.add_argument('--save_dir', help='directory to save models')


args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def test(net, dataloader):
    global best_acc
    criterion = nn.CrossEntropyLoss()
    net.train()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).long().sum().item()

    # Save checkpoint when best model
    acc = 100.*correct/total
    test_loss = test_loss / total
    print("\n| Validation Loss: %.4f Acc@1: %.2f%%" %(test_loss, acc))
    return acc


if __name__ == '__main__':
    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif (args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)


    # Model
    print('\n[Phase 2] : Model setup')
    print('| Building net type [' + args.net + ']...')
    if args.net == 'resnet34':
        net = ResNet(34, num_classes, 0.5)
    elif args.net == 'densenet':
        net = DenseNet3(100, num_classes, 12, 0.5, True, 0.2)
    elif args.net == 'vgg16':
        net = VGGNet(num_classes, 0.5, False, 2048, True)
    else:
        print('Error : Network should be either [ResNet34]')
        sys.exit(0)

    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['model'])
    net.to(device)

    avg = 0

    for i in range(10):
        avg += test(net, testloader)

    print(avg / 10)