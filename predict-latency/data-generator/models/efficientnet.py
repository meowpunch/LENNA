'''EfficientNet in PyTorch.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        # SE layers
        self.fc1 = nn.Conv2d(out_planes, out_planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(out_planes//16, out_planes, kernel_size=1)

    def forward(self, x):
        # print(x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.size())
        out = F.relu(self.bn2(self.conv2(out)))
        # print(out.size())
        out = self.bn3(self.conv3(out))
        # (out.size())
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        # print(out.size())
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(cfg[-1][1], num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            # print("strids: ", strides)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, count):
        # if count is 0:
        #     print("1.", x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        # if count is 0:
        #     print("2.",out.size())
        #     # print("...머여")
        out = self.layers(out)
        # print(out.size(0))
        # if count is 0:
        #     print("3.",out.size())
        #     print(out.size(0), out.size(1), out.size(2), out.size(3))
        out = out.view(out.size(0)*out.size(2)*out.size(3), -1)
        # if count is 0:
        #     print("4.",out.size())
        out = self.linear(out)
        return out


def EfficientNetB0(cfg):
    # (expansion, out_planes, num_blocks, stride)

    # cfg = [(1,  16, 1, 1),
    #        (6,  24, 2, 1),
    #        (6,  40, 2, 4),
    #        (6,  80, 3, 2),
    #        (6, 112, 3, 4),
    #        (6, 192, 4, 1),
    #        (6, 320, 1, 1)]

    # print("in efficientnet")
    # print(cfg)
    return EfficientNet(cfg)


"""
    # 안씀
    def test():
        net = EfficientNetB0()
        x = torch.randn(2, 3, 32, 32)
        y = net(x)
        print(y.shape)
    
    
    # test()
"""

