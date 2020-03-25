import os
import sys

import torch

from data_pipeline.data_generator import DataGenerator
from utils.logger import init_logger


class DataPipeline:
    def __init__(self):
        self.logger = init_logger()

        self.data_path = "training_data"
        self.file_name = "training_data"

        self.X = None
        self.y = None

    def process(self):
        """
        return: pandas DataFrame and save file
        """
        self.logger.info(("main start pid: %s" % (os.getpid())))

        # parallel process
        for i in range(4):
            child_pid = os.fork()

            if child_pid == 0:
                self.logger.info("child %s" % (os.getpid()))
                dg = DataGenerator()
                self.X, self.y = dg.execute()

                self.save_file()

        # wait children
        if os.getpid() is not 0:
            status = os.wait()
            self.logger.info("\nIn parent process-")
            self.logger.info("Terminated child's process id:", status[0])
            self.logger.info("Signal number that killed the child process:", status[1])
        return

    def save_file(self):
        if os.path.isfile('training_data/' + self.file_name) is True:
            f = open('training_data/' + self.file_name, "a")
        else:
            f = open('training_data/' + self.file_name, "w")

        f.writelines(self.X)
        f.write(', ')
        f.write(self.y)
        f.write('\n')

        f.close()
        sys.exit(0)


def main():
    """
        TODO:
        0: SuperDartsRecastingNet
        1: LennaNet
    :return:
    """
    pipeline = DataPipeline()
    pipeline.process()


if __name__ == '__main__':
    main()
