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

parser = argparse.ArgumentParser(description='PyTorch CIFAR')

parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--num_epochs', default=300, type=int, help='number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net', default='resnet34', type=str, help='model')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--stoch_depth', default=1, type=float, help='Stochastic depth; 1 means off (default=1)')

parser.add_argument('--distill_from', default=1, type=int, help='epoch to start distillation')
parser.add_argument('--distill', type=float, default=0, metavar='M', help='factor of distill loss (default: 0.1, off if <=0)')
parser.add_argument('--temp', type=float, default=1, metavar='M', help='temperature for distillation (default: 7)')

# set training session
parser.add_argument('--seed', help='pytorch random seed', default=1, type=int)
parser.add_argument('--save_dir', help='directory to save models')


args = parser.parse_args()
best_acc = 0
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device('cpu')

if args.stoch_depth == 1:
    model_name = '{}_{}_s1_without_sd_seed{}'.format(args.net, args.dataset, args.seed)
else:
    model_name = '{}_{}_s1_t{}_d{}_seed{}'.format(args.net, args.dataset, args.temp, args.distill, args.seed)
log_file_name = os.path.join(args.save_dir, 'Log_{}.txt'.format(model_name))
log_file = open(log_file_name, 'w')


# Training
def train(net, dataloader, optimizer, epoch):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print('\n=> [%s] Training Epoch #%d, lr=%.4f' %(model_name, epoch, cf.learning_rate(args.lr, epoch)))
    log_file.write('\n=> [%s] Training Epoch #%d, lr=%.4f\n' %(model_name, epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # obtain soft_target by forwarding data in test mode
        if epoch >= args.distill_from and args.distill > 0:
            with torch.no_grad():
                net.eval()
                soft_target = net(inputs)

        net.train()
        optimizer.zero_grad()
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss

        # compute distillation loss
        if epoch >= args.distill_from and args.distill > 0:
            heat_output = outputs / args.temp
            heat_soft_target = soft_target / args.temp

            distill_loss = F.kl_div(F.log_softmax(heat_output, 1), F.softmax(heat_soft_target), size_average=False) / targets.size(0) * (args.temp*args.temp)
            loss = loss + args.distill * distill_loss

        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.detach(), 1)
        total += targets.size(0)
        correct += predicted.eq(targets.detach()).long().sum().item()

        if math.isnan(loss.item()):
            print('@@@@@@@nan@@@@@@@@@@@@')
            log_file.write('@@@@@@@@@@@nan @@@@@@@@@@@@@\n')
            sys.exit(0)

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1,
                    (len(trainset)//args.bs)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()
    log_file.write('| Epoch [%3d/%3d] \t\tLoss: %.4f Acc@1: %.3f%%'
                     % (epoch, args.num_epochs, loss.item(), 100. * correct / total))


def test(net, dataloader, epoch):
    global best_acc
    criterion = nn.CrossEntropyLoss()
    net.eval()
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
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss, acc))
    log_file.write("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%\n" %(epoch, test_loss, acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        log_file.write('| Saving Best model...\t\t\tTop1 = %.2f%%\n' %(acc))
        save_name = os.path.join(args.save_dir, '{}.pth'.format(model_name))
        checkpoint = dict()
        checkpoint['model'] = net.state_dict()
        checkpoint['model_name'] = model_name
        checkpoint['seed'] = args.seed
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, save_name)
        best_acc = acc


def test_trainset(net, dataloader, epoch):
    criterion = nn.CrossEntropyLoss()
    net.eval()
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
    print("\n| Evaluation Trainset Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss, acc))
    log_file.write("\n| Evaluation Trainset Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%\n" %(epoch, test_loss, acc))


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    log_file.write(str(args))
    log_file.write('\n')

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif (args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)


    # Model
    print('\n[Phase 2] : Model setup')
    print('| Building net type [' + args.net + ']...')
    if args.net == 'resnet34':
        net = ResNet(34, num_classes, args.stoch_depth)
    else:
        print('Error : Network should be either [ResNet34]')
        sys.exit(0)

    net.init_weights()
    net.to(device)

    # Training
    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))

    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, 1), momentum=0.9, weight_decay=1e-4)

    elapsed_time = 0
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        set_learning_rate(optimizer, cf.learning_rate(args.lr, epoch))
        train(net, trainloader, optimizer, epoch)
        test_trainset(net, trainloader, epoch)
        test(net, testloader, epoch)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' %(cf.get_hms(elapsed_time)))
        log_file.write('| Elapsed time : %d:%02d:%02d\n' %(cf.get_hms(elapsed_time)))
        log_file.flush()

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' %(best_acc))
    log_file.write('* Test results : Acc@1 = %.2f%%\n' %(best_acc))
