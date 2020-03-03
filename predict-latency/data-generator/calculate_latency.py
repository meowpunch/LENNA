from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


import os
import argparse

from models import *


# bucket 안 row 개수
num_data = 100

'''

# (expansion, out_planes, num_blocks, stride)
cfg = [(1,  16, 1, 2),
       (6,  24, 2, 1),
       (6,  40, 2, 2),
       (6,  80, 3, 2),
       (6, 112, 3, 1),
       (6, 192, 4, 2),
       (6, 320, 1, 2)]

num_types: 1~15 integer  위의 예제는 7임
---
expansion: 1~6 integer
out_planes: 16~320 integer
num_blocks: sum is 15 and each value is integer
stride: 1 or 2 ~ 더 커질 수 있음.

assume that depth scale is fixed.

'''


def calculate_latency(cfg):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test -accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    net = EfficientNetB0(cfg)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # print(net)

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    each_y_sum = 0
    count = 0

    # estimate latency of infer
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs)
            # loss = criterion(outputs, targets)

            flag=0
            with torch.autograd.profiler.profile(use_cuda=True) as prof:  # with torchprof.Profile(net, use_cuda=True) as prof:  #
                outputs = net(inputs,count)
                # print(prof)
            print(prof.self_cpu_time_total)
            if count > 2:
                each_y_sum += prof.self_cpu_time_total  # latency
            count += 1

            if count == 7:
                break

    print("over")
    print(each_y_sum, count)
    # 1개 버리고 4개해서 평균냄. test경우는 train과 다르게 처음부터 비슷한 값을 지님.
    latency = each_y_sum/(count-3)

    print(latency)

    return latency