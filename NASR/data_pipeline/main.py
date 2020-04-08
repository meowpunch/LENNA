import os
import sys
from multiprocessing.pool import Pool
import multiprocessing

import torch
import torchvision
from torchvision.transforms import transforms

from data_pipeline.data_generator import DataGenerator
from util.daemon import MyPool
from util.logger import init_logger


class DataPipeline:
    def __init__(self, arg):
        self.logger = init_logger()

        # constant
        self.destination = "training_data/data{postfix}".format(
            postfix=arg
        )

        self.X = None
        self.y = None

    def process(self, load):
        """
            TODO: return value
        """
        shadow = os.fork()
        latency_list = None
        if shadow == 0:
            dg = DataGenerator()
            self.X, self.y, latency_list = dg.process(load)
            # print(latency_list)
            self.logger.info("X: {X}, y: {y}".format(
                X=self.X, y=self.y
            ))

            self.save_file()
            sys.exit()
        else:
            self.logger.info("%s worker got shadow %s" % (os.getpid(), shadow))

        pid, status = os.waitpid(shadow, 0)
        self.logger.info("wait returned, pid = %d, status = %d" % (pid, status))
        return latency_list

    def save_file(self):
        if os.path.isfile(self.destination) is True:
            f = open(self.destination, "a")
        else:
            f = open(self.destination, "w")

        f.writelines(' '.join(list(map(lambda x: str(x), self.X))))
        f.write(', ')
        f.write(str(self.y))
        f.write('\n')
        f.close()


class Worker:
    def __init__(self, load):
        self.load = load

    def __call__(self, x):
        DataPipeline(x).process(self.load)


def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = torchvision.datasets.CIFAR10(root='../Recasting_ver/data', train=False,
                                            download=True, transform=transform)
    return torch.utils.data.DataLoader(test_set, batch_size=4,
                                       shuffle=False, num_workers=2)


def main():

    init_logger().info("director id: %s" % (os.getpid()))
    p_num = 1

    with MyPool(p_num) as p:
        p.map(Worker(load_dataset()), range(p_num))


if __name__ == '__main__':
    main()


