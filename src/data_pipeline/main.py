import os

import pandas as pd

from data_pipeline.core import DataPipeline
from multiprocessing import Process, Lock
from util.daemon import MyPool
from util.dataset import load_dataset
from util.logger import init_logger


class Worker:
    def __init__(self, load, destination, outer_loop, inner_loop):
        self.load = load
        self.destination = destination
        self.out_loop = outer_loop
        self.in_loop = inner_loop

    def __call__(self, x):
        DataPipeline(x, self.destination).process(
            load=self.load, o_loop=self.out_loop, shadow=True, i_loop=self.in_loop
        )


def worker(index, load, destination, lock, outer_loop, inner_loop):
    DataPipeline(index, destination, lock).process(load, o_loop=outer_loop, shadow=True, i_loop=inner_loop)


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


def parallel(destination, outer_loop, inner_loop, p_num=4):
    lock = Lock()
    procs = []

    # generate child process
    for i in range(p_num):
        proc = Process(
            target=worker, args=(i, load_dataset(batch_size=32), destination, lock, outer_loop, inner_loop)
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def single(destination, o_loop=250, i_loop=10, b_type=None, in_ch=None):
    DataPipeline(0, destination, None).process(
        load_dataset(batch_size=64), o_loop=o_loop, shadow=True, i_loop=i_loop, b_type=b_type,in_ch=in_ch
    )


def main(arg="parallel"):
    destination = "training_data/data"
    if arg is "parallel":
        logger = init_logger()
        logger.info("director id: %s" % (os.getpid()))
        parallel(destination=destination, outer_loop=2, inner_loop=1, p_num=4)
    else:
        single(destination=destination)


if __name__ == '__main__':
    main("single")
