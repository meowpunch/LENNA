from generate_data import generate_data

import os

import torchvision
import torchvision.transforms as transforms
import torch

def dataload():


    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    return testloader


if __name__ == '__main__':
    file_name = './training_data00'
    test_ld = dataload()

    for j in range(10):
        # make_bucket()
        if os.path.isfile(file_name) is True:
            f = open(file_name, "a")
        else:
            f = open(file_name, "w")

        for i in range(100):
            cfg, target, valid = generate_data(test_ld)
            # print("in main")
            print("count: ", i)
            # print(valid)
            # print(cfg)
            # print(target)
            if valid is 3:
                f.writelines(cfg)
                f.write(', ')
                f.write(target)
                f.write('\n')

        f.close()
        # generate_data()
