import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys
import numpy as np
from torch.nn import init


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=1):
        super(BasicBlock, self).__init__()
        self.p = p
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #if self.training and torch.rand(1)[0] >= self.p:
        if self.training and np.random.binomial(1, self.p, 1)[0] == 0:
            return self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.training:
            out = out / self.p
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, p=1):
        super(Bottleneck, self).__init__()
        self.p = p
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if self.training and np.random.binomial(1, self.p, 1)[0] == 0:
            return self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.training:
            out = out / self.p
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, depth, num_classes, p=1): # p==1: no stochastic depth
        super(ResNet, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks, 0, stride=1, p=p)
        self.layer2 = self._make_layer(block, 128, num_blocks, 1, stride=2, p=p)
        self.layer3 = self._make_layer(block, 256, num_blocks, 2, stride=2, p=p)
        self.layer4 = self._make_layer(block, 512, num_blocks, 3, stride=2, p=p)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)
        #self.apply(conv_init)

    def _make_layer(self, block, planes, num_blocks, block_idx, stride, p):
        strides = [stride] + [1]*(num_blocks[block_idx]-1)
        layers = []

        start_idx = sum(num_blocks[:block_idx])
        n_blocks = sum(num_blocks)

        for i, stride in enumerate(strides):
            block_p = (1-(float(start_idx+i)/(n_blocks)*(1-p)))
            layers.append(block(self.in_planes, planes, stride, block_p))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=ResNet(50, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
