import os
from multiprocessing import Pool

import torch
import torchvision
from torchvision.transforms import transforms

from data_pipeline.core import DataPipeline
from util.logger import init_logger


class Worker:
    def __init__(self, load, destination):
        self.load = load
        self.destination = destination

    def __call__(self, x):
        DataPipeline(x, self.destination).process(self.load)


def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = torchvision.datasets.CIFAR10(root='../Recasting_ver/data', train=False,
                                            download=True, transform=transform)
    return torch.utils.data.DataLoader(test_set, batch_size=4,
                                       shuffle=False, num_workers=2)


def save(destination, num):
    if os.path.isfile(destination) is True:
        f = open(destination, "a")
    else:
        f = open(destination, "w")

    for i in range(num):
        with open(destination + i, 'rb') as pf:
            f.write(pf.read())
            pf.close()

    f.close()


def main():
    init_logger().info("director id: %s" % (os.getpid()))
    destination = "training_data/data"
    p_num = 3

    with Pool(p_num) as pool:
        pool.map(Worker(
            load=load_dataset(),
            destination=destination
        ), range(p_num))

    pool.close()
    pool.join()

    # save(destination=destination, num=p_num)


if __name__ == '__main__':
    main()
