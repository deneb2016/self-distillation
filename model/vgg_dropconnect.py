import torch
import torch.nn as nn
import math
from model.dropconnect import DropConnect
import torchvision.models


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'C4': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'C5': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


class VGGNetDropConnect(nn.Module):
    def __init__(self, num_classes, drop_p=0, drop_last_only=False, feat_dim=1024):
        super(VGGNetDropConnect, self).__init__()
        self.features = make_layers(cfg['C5'], True)

        layers = list()
        if drop_p > 0 and not drop_last_only:
            layers.append(DropConnect(512, feat_dim, drop_p))
        else:
            layers.append(nn.Linear(512, feat_dim))
        layers.append(nn.ReLU(True))

        if drop_p > 0:
            print('@@@@@@@@@@@ use dropconnect @@@@@@@@@@')
            layers.append(DropConnect(feat_dim, feat_dim, drop_p))
        else:
            layers.append(nn.Linear(feat_dim, feat_dim))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(feat_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, DropConnect):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        N = x.size(0)
        out = self.features(x)
        out = out.view(N, -1)
        out = self.classifier(out)
        return out
