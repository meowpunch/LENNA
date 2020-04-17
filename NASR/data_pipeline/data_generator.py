from functools import reduce

import numpy as np

from data_pipeline.latency_estimator import LatencyEstimator
from util.logger import init_logger


class DataGenerator:
    def __init__(self):
        """
            block type: 0 -> reduction , 1-> normal
            input_channel: 1~1000
            num_layers: fix 5 or 6
        """
        self.logger = init_logger()

        # X
        np.random.seed()
        self.block_type = 1
        self.input_channel = 512
        self.num_layers = 1
        self.arch_params = None

        self.logger.info("{} {} {}".format(
            self.block_type, self.input_channel, self.num_layers
        ))

        # y
        self.latency = None

    def serialize_x(self):
        return np.append(np.array([
            self.block_type, self.input_channel, self.num_layers,
        ]), reduce(
            lambda a, b: np.append(a, b), self.arch_params
        ))

    def process(self, load):
        """
        return: X, y, latency_list
        """
        # get latency and arch_params (randomly chosen in normal distribution)
        self.arch_params, self.latency, latency_list = LatencyEstimator(
            block_type=self.block_type,
            input_channel=self.input_channel,
            num_layers=self.num_layers,
            dataset=load
        ).execute()

        return self.serialize_x(), self.latency, latency_list
