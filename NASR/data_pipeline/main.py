import os

from data_pipeline.core import DataPipeline
from util.daemon import MyPool
from util.dataset import load_dataset
from util.logger import init_logger


class Worker:
    def __init__(self, load, destination):
        self.load = load
        self.destination = destination

    def __call__(self, x):
        DataPipeline(x, self.destination).process(self.load)


def collect_data(destination, num):
    if os.path.isfile(destination) is True:
        f = open(destination, "ab")
    else:
        f = open(destination, "wb")

    # combine
    for i in range(num):
        with open(destination + str(i), 'rb') as pf:
            f.write(pf.read())
            pf.close()
    f.close()

    # delete
    for i in range(num):
        if os.path.exists(destination + str(i)):
            os.remove(destination + str(i))


def parallel(destination):
    logger = init_logger()
    logger.info("director id: %s" % (os.getpid()))

    p_num = 3

    with MyPool(p_num) as pool:
        pool.map(Worker(
            load=load_dataset(),
            destination=destination
        ), range(p_num))

    pool.close()
    pool.join()

    collect_data(destination=destination, num=p_num)
    logger.info("success to collect data into '{dest}'".format(dest=destination))


def single(destination):
    DataPipeline(0, destination).process(load_dataset(), arg=True)


def main(arg="parallel"):
    destination = "training_data/data"
    if arg is "parallel":
        parallel(destination=destination)
    else:
        single(destination=destination)


if __name__ == '__main__':
    main("single")
