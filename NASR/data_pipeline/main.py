import os

import pandas as pd

from data_pipeline.core import DataPipeline
from multiprocessing import Process, Lock
from util.daemon import MyPool
from util.dataset import load_dataset
from util.logger import init_logger


class Worker:
    def __init__(self, load, destination):
        self.load = load
        self.destination = destination

    def __call__(self, x):
        DataPipeline(x, self.destination).process(self.load)


def worker(index, load, destination, lock):
    DataPipeline(index, destination, lock).process(load)


def collect_df(destination, num):
    # collect df from csv files
    combined_df = pd.concat([pd.read_csv(destination + str(i)) for i in range(num)], axis=0)

    # save
    if os.path.isfile(destination) is True:
        combined_df.to_csv(destination, mode='w', index=False, header=True)
    else:
        combined_df.to_csv(destination, mode='a', index=False, header=False)

    # check
    init_logger().info("final saved df's tail 5: \n{df}".format(df=pd.read_csv(destination).tail(5)))

    # delete
    for i in range(num):
        if os.path.exists(destination + str(i)):
            os.remove(destination + str(i))

    return 0


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


def pool_parallel(destination, p_num=4):
    # generate child process
    with MyPool(p_num) as pool:
        pool.map(Worker(
            load=load_dataset(batch_size=64),
            destination=destination
        ), range(p_num))

    # wait all child process to work done
    pool.close()
    pool.join()

    # collect data
    collect_df(destination=destination, num=p_num)
    init_logger().info("success to collect data into '{dest}'".format(dest=destination))


def parallel(destination, p_num=4):
    lock = Lock()
    procs = []

    # generate child process
    for i in range(p_num):
        proc = Process(
            target=worker, args=(i, destination, load_dataset(batch_size=64), lock)
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def single(destination):
    DataPipeline(0, destination).process(load_dataset(batch_size=64))


def main(arg="parallel"):
    destination = "training_data/data"
    if arg is "parallel":
        logger = init_logger()
        logger.info("director id: %s" % (os.getpid()))
        parallel(destination=destination, p_num=4)
    else:
        single(destination=destination)


if __name__ == '__main__':
    main("parallel")
