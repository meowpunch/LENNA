import os
import sys
from multiprocessing.pool import Pool
import multiprocessing

import torch

from data_pipeline.data_generator import DataGenerator
from utils.daemon import MyPool
from utils.logger import init_logger


class DataPipeline:
    def __init__(self, arg):
        self.logger = multiprocessing.get_logger()

        # constant
        self.destination = "training_data/data{postfix}".format(
            postfix=arg
        )

        self.X = None
        self.y = None

    def process(self):
        """
            TODO: return value
        return: pandas DataFrame and save file
        """
        self.logger.info(("main start pid: %s" % (os.getpid())))

        # parallel process
        # for i in range(4):
        shadow = os.fork()

        if shadow == 0:
            self.logger.info("child %s" % (os.getpid()))
            dg = DataGenerator()
            self.X, self.y = dg.execute()

            self.save_file()
            sys.exit()
        else:
            self.logger.info("light(%s) got shadow:%s" % (os.getpid(), shadow))

        pid, status = os.waitpid(shadow, 0)
        print("wait returned, pid = %d, status = %d" % (pid, status))

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


def work(pipeline):
    pipeline.process()


def main():

    p_num = 3
    tasks = [DataPipeline(i) for i in range(p_num)]

    with MyPool(p_num) as p:
        p.map(work, tasks)


if __name__ == '__main__':
    main()


