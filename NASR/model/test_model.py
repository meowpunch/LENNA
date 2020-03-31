import torch.nn as nn


class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1, self).__init__()
        self.choices = nn.ModuleDict({
            '5x5 conv': nn.Conv2d(1, 1, 5),
            '3x3 conv': nn.Conv2d(1, 1, 3),
            '7x7 conv': nn.Conv2d(1, 1, 7)
        })

    def forward(self, x):
        print(x.shape)
        x = self.choices['5x5 conv'](x)
        print(x.shape)
        x = self.choices['3x3 conv'](x)
        print(x.shape)
        x = self.choices['7x7 conv'](x)
        print(x.shape)
        return x


class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.choices = nn.ModuleDict({
            '3x3 conv1': nn.Conv2d(1, 1, 3),
            '3x3 conv2': nn.Conv2d(1, 1, 3),
            '3x3 conv3': nn.Conv2d(1, 1, 3),
            '3x3 conv4': nn.Conv2d(1, 1, 3),
            '3x3 conv5': nn.Conv2d(1, 1, 3),
        })

    def forward(self, x):
        print(x.shape)
        x = self.choices['3x3 conv1'](x)
        print(x.shape)
        x = self.choices['3x3 conv2'](x)
        print(x.shape)
        x = self.choices['3x3 conv3'](x)
        print(x.shape)
        x = self.choices['3x3 conv4'](x)
        print(x.shape)
        x = self.choices['3x3 conv5'](x)
        print(x.shape)
        return x


class MyModel3(nn.Module):
    def __init__(self):
        super(MyModel3, self).__init__()
        self.choices = nn.ModuleDict({
            '7x7 conv': nn.Conv2d(1, 1, 7),
            '3x3 conv': nn.Conv2d(1, 1, 3),
            '5x5 conv': nn.Conv2d(1, 1, 5)
        })

    def forward(self, x):
        print(x.shape)
        x = self.choices['7x7 conv'](x)
        print(x.shape)
        x = self.choices['3x3 conv'](x)
        # print(x.shape)
        x = self.choices['5x5 conv'](x)
        # print(x.shape)
        return x


class Parallel(nn.Module):
    def __init__(self):
        super(Parallel, self).__init__()
        self.choices = nn.ModuleDict({
            '3x3 conv1': nn.Conv2d(1, 1, 3),
            '3x3 conv2': nn.Conv2d(1, 1, 3),
            '3x3 conv3': nn.Conv2d(1, 1, 3),
            '3x3 conv4': nn.Conv2d(1, 1, 3),
            '3x3 conv5': nn.Conv2d(1, 1, 3),
        })

    def forward(self, x):
        print(x.shape)
        return self.choices['3x3 conv1'](x) + self.choices['3x3 conv2'](x) \
               + self.choices['3x3 conv3'](x) + self.choices['3x3 conv4'](x) + self.choices['3x3 conv5'](x)


class Reduction(nn.Module):
    def __init__(self):
        super(Reduction, self).__init__()
        self.choices = nn.ModuleDict({
            '5x5 conv1': nn.Conv2d(1, 1, 5),
            '3x3 conv2': nn.Conv2d(1, 1, 3),
            '3x3 conv3': nn.Conv2d(1, 1, 3),
            '3x3 conv4': nn.Conv2d(1, 1, 3),
            '3x3 conv5': nn.Conv2d(1, 1, 3),
            '3x3 conv6': nn.Conv2d(1, 1, 3)
        })

    def forward(self, x):
            x1 = self.choices['5x5 conv1'](x)
            x2 = self.choices['3x3 conv2'](x) + self.choices['3x3 conv3'](x1)
            x3 = self.choices['3x3 conv4'](x1) + self.choices['3x3 conv5'](x)
            return x2 + x3 + self.choices['3x3 conv6']


