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

    def process(self, load, parallel=True, num_rows=100):
        """
        """
        i = 0
        if parallel:
            while i < 2500:
                shadow = os.fork()
                latency_list = None
                if shadow == 0:
                    dg = DataGenerator(sub_pid=self.sub_pid)
                    self.df = dg.process(load=load, num_rows=num_rows)

                    self.save_to_csv()

                    sys.exit()
                else:
                    self.logger.info("%s worker got shadow %s" % (os.getpid(), shadow))

                pid, status = os.waitpid(shadow, 0)
                self.logger.info("wait returned, pid = %d, status = %d" % (pid, status))

                # TODO: increase 1 to i on shadow.
                i = i + 1
                self.logger.info("------------------------ {}*{} rows ------------------------".format(i, num_rows))
            return 0
        else:
            dg = DataGenerator()
            self.df = dg.process(load=load, num_rows=1)
            self.save_to_csv()
            return 0

    def save_to_csv(self):
        # lock
        if self.lock:
            self.lock.acquire()

        # save
        if os.path.isfile(self.destination) is True:
            self.df.to_csv(self.destination, mode='a', index=False, header=False)
        else:
            self.df.to_csv(self.destination, mode='w', index=False, header=True)
        self.logger.info("success to save data in '{dest}'".format(dest=self.destination))

        # unlock
        if self.lock:
            self.lock.release()