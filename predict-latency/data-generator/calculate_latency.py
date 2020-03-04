from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *


def calculate_latency(cfg, testloader):


    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

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

            flag = 0
            with torch.autograd.profiler.profile(
                    use_cuda=True) as prof:  # with torchprof.Profile(net, use_cuda=True) as prof:  #
                outputs = net(inputs, count)
                # print(prof)
            print(prof.self_cpu_time_total)
            if count > 2:
                each_y_sum += prof.self_cpu_time_total  # latency
            count += 1

            if count == 7:
                break

    print("each_y_sum, count")
    print(each_y_sum, count)
    # 1개 버리고 4개해서 평균냄. test 경우는 train 과 다르게 처음부터 비슷한 값을 지님.
    latency = each_y_sum / (count - 3)

    print(latency)
    return latency
