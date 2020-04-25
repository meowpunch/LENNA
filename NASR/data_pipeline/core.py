import os
import sys

from data_pipeline.data_generator import DataGenerator
from util.logger import init_logger


class DataPipeline:
    def __init__(self, arg, destination):
        self.logger = init_logger()

        # constant
        self.destination = destination + str(arg)

        self.X = None
        self.y = None

    def process(self, load, arg=True):
        """
            TODO: not produce shadow process.
        """
        shadow = os.fork()
        latency_list = None
        if shadow == 0:
            dg = DataGenerator()
            self.X, self.y = dg.process(load)
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
        return 0

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

        self.logger.info("success to save data in '{dest}'".format(dest=self.destination))
