import os
import sys

from data_pipeline.data_generator import DataGenerator
from util.logger import init_logger


class DataPipeline:
    def __init__(self, idx, destination, lock):
        self.logger = init_logger()
        self.sub_pid = idx
        self.lock = lock

        # constant
        self.destination = destination  # + str(idx)

        self.df = None

    def process(self, load, shadow=True):
        """
            TODO: not produce shadow process.
        """
        if shadow:
            shadow = os.fork()
            latency_list = None
            if shadow == 0:
                dg = DataGenerator(sub_pid=self.sub_pid)
                self.df = dg.process(load=load, num_rows=1)

                self.save_file()
                sys.exit()
            else:
                self.logger.info("%s worker got shadow %s" % (os.getpid(), shadow))

            pid, status = os.waitpid(shadow, 0)
            self.logger.info("wait returned, pid = %d, status = %d" % (pid, status))
            return 0
        else:
            dg = DataGenerator()
            self.df = dg.process(load=load, num_rows=1)
            self.save_file()
            return 0

    def save_file(self):
        # lock
        self.lock.acquire()

        # save
        if os.path.isfile(self.destination) is True:
            self.df.to_csv(self.destination, mode='w', index=False, header=True)
        else:
            self.df.to_csv(self.destination, mode='a', index=False, header=False)
        self.logger.info("success to save data in '{dest}'".format(dest=self.destination))

        # unlock
        self.lock.release()