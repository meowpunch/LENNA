import time
from functools import reduce

import pandas as pd
import numpy as np
import torch.nn as nn
from Recasting_ver.modules.layers import *

'''
Output Size = (W - F + 2P) / S + 1
W: input_volume_size
F: kernel_size
P: padding_size
S: strides
'''

# constant
padding_5 = (5 - 1) // 2
padding_3 = (3 - 1) // 2
padding_7 = (7 - 1) // 2


class MyModel1(MyModule):
    def __init__(self):
        super(MyModel1, self).__init__()
        self.modules = {
            '5x5 conv': nn.Conv2d(16, 16, 5, padding=padding_5),
            '3x3 conv': nn.Conv2d(16, 16, 3, padding=padding_3),
            '7x7 conv': nn.Conv2d(16, 16, 7, padding=padding_7)
        }
        self.choices = nn.ModuleDict(self.modules)

    def forward(self, x):
        t0 = time.time()

        x1 = self.choices['5x5 conv'](x)
        t1 = time.time()

        x2 = self.choices['3x3 conv'](x1)
        t2 = time.time()

        x3 = self.choices['7x7 conv'](x2)
        t3 = time.time()

        self.latency_list.append(
            # 5x5, 3x3, 7x7, total
            self.unit_transform([t1 - t0, t2 - t1, t3 - t2, t3 - t0])
        )
        self.size_list = [x, x1, x2, x3]
        return x3


class MyModel2(MyModule):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.modules = {
            '3x3 conv1': nn.Conv2d(16, 32, 3, padding=padding_3),
            '3x3 conv2': nn.Conv2d(32, 64, 3, padding=padding_3),
            '3x3 conv3': nn.Conv2d(64, 32, 3, padding=padding_3),
            '3x3 conv4': nn.Conv2d(32, 16, 3, padding=padding_3),
            '3x3 conv5': nn.Conv2d(16, 1, 3, padding=padding_3)
        }
        self.choices = nn.ModuleDict(self.modules)

    def forward(self, x):
        t0 = time.time()

        x1 = self.choices['3x3 conv1'](x)
        t1 = time.time()

        x2 = self.choices['3x3 conv2'](x1)
        t2 = time.time()

        x3 = self.choices['3x3 conv3'](x2)
        t3 = time.time()

        x4 = self.choices['3x3 conv4'](x3)
        t4 = time.time()

        x5 = self.choices['3x3 conv5'](x4)
        t5 = time.time()

        self.latency_list.append(
            # 5x5, 3x3, 7x7, total
            self.unit_transform([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0])
        )
        self.size_list = [x, x1, x2, x3, x4, x5]
        return x5


class MyModel3(MyModule):
    def __init__(self):
        super(MyModel3, self).__init__()
        self.modules = {
            '7x7 conv': nn.Conv2d(16, 16, 7, padding=padding_7),
            '3x3 conv': nn.Conv2d(16, 16, 3, padding=padding_3),
            '5x5 conv': nn.Conv2d(16, 16, 5, padding=padding_5)
        }
        self.choices = nn.ModuleDict(self.modules)

    def forward(self, x):
        t0 = time.time()

        x1 = self.choices['7x7 conv'](x)
        t1 = time.time()

        x2 = self.choices['3x3 conv'](x1)
        t2 = time.time()

        x3 = self.choices['5x5 conv'](x2)
        t3 = time.time()

        self.latency_list.append(
            # 5x5, 3x3, 7x7, total
            self.unit_transform([t1 - t0, t2 - t1, t3 - t2, t3 - t0])
        )
        self.size_list = [x, x1, x2, x3]
        return x3


class MyModel4(MyModule):
    def __init__(self):
        super(MyModel4, self).__init__()
        self.modules = {
            '7x7 conv': nn.Conv2d(1, 1, 7, padding=padding_7),
            '3x3 conv': nn.Conv2d(1, 1, 3, padding=padding_3),
            '5x5 conv': nn.Conv2d(1, 1, 5, padding=padding_5)
        }
        self.choices = nn.ModuleDict(self.modules)

    def forward(self, x):
        t0 = time.time()

        x1 = self.choices['7x7 conv'](x)
        t1 = time.time()

        x2 = self.choices['3x3 conv'](x1)
        t2 = time.time()

        x3 = self.choices['5x5 conv'](x2)
        t3 = time.time()

        self.latency_list.append(
            # 5x5, 3x3, 7x7, total
            self.unit_transform([t1 - t0, t2 - t1, t3 - t2, t3 - t0])
        )
        self.size_list = [x, x1, x2, x3]
        return x3


class Parallel1(MyModule):
    def __init__(self):
        super(Parallel1, self).__init__()
        self.modules = {
            '7x7 conv': nn.Conv2d(1, 1, 7, padding=padding_7),
            '3x3 conv': nn.Conv2d(1, 1, 3, padding=padding_3),
            '5x5 conv': nn.Conv2d(1, 1, 5, padding=padding_5)
        }
        self.choices = nn.ModuleDict(self.modules)

    def forward(self, x):
        t0 = time.time()

        x_out = self.choices['7x7 conv'](x) + \
                self.choices['3x3 conv'](x) + \
                self.choices['5x5 conv'](x)
        t1 = time.time()

        self.latency_list.append(
            # total
            self.unit_transform([np.nan, np.nan, np.nan, t1 - t0])
        )
        self.size_list = [x, x_out]
        return x_out


class Parallel2(MyModule):
    def __init__(self):
        super(Parallel2, self).__init__()
        self.modules = {
            '7x7 conv': nn.Conv2d(1, 1, 7, padding=padding_7),
            '3x3 conv': nn.Conv2d(1, 1, 3, padding=padding_3),
            '5x5 conv': nn.Conv2d(1, 1, 5, padding=padding_5),
        }
        self.choices = nn.ModuleDict(self.modules)

    def forward(self, x):
        t0 = time.time()

        x1 = self.choices['7x7 conv'](x)
        t1 = time.time()

        x2 = self.choices['3x3 conv'](x)
        t2 = time.time()

        x3 = self.choices['5x5 conv'](x)
        t3 = time.time()

        self.latency_list.append(
            # 5x5, 3x3, 7x7, total
            self.unit_transform([t1 - t0, t2 - t1, t3 - t2, t3 - t0])
        )
        self.size_list = [x, x1, x2, x3]
        return x1 + x2 + x3


class Reduction(MyModule):
    def __init__(self):
        super(Reduction, self).__init__()
        self.modules = {
            '3x3 conv1': nn.Conv2d(1, 1, 3, padding=padding_3),
            '3x3 conv2': nn.Conv2d(1, 1, 3, padding=padding_3),
            '3x3 conv3': nn.Conv2d(1, 1, 3, padding=padding_3),
            '3x3 conv4': nn.Conv2d(1, 1, 3, padding=padding_3),
            '3x3 conv5': nn.Conv2d(1, 1, 3, padding=padding_3),
            '3x3 conv6': nn.Conv2d(1, 1, 3, padding=padding_3)
        }
        self.choices = nn.ModuleDict(self.modules)

    def forward(self, x):
        t0 = time.time()

        x1 = self.choices['3x3 conv1'](x)
        t1 = time.time()

        x2 = self.choices['3x3 conv2'](x) + self.choices['3x3 conv3'](x1)
        t2 = time.time()

        x3 = self.choices['3x3 conv4'](x1) + self.choices['3x3 conv5'](x)
        t3 = time.time()

        x4 = self.choices['3x3 conv6'](x)
        t4 = time.time()

        self.latency_list.append(
            # 5x5, 3x3, 7x7, total
            self.unit_transform([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, t4 - t0])
        )
        self.size_list = [x, x1, x2, x3, x4]
        return x2 + x3 + x4
