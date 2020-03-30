from functools import reduce

import numpy as np

from data_pipeline.latency_estimator import LatencyEstimator
from utils.logger import init_logger


class DataGenerator:
    def __init__(self):
        """
            block type: 0 -> reduction , 1-> normal
            input_channel: 1~1000
            output_channel: 1~1000
            num_layers: fix 5 or 6
        """
        self.logger = init_logger()

        # X
        np.random.seed()
        self.block_type = np.random.randint(0, 2)
        self.input_channel = np.random.randint(1, 1001)
        # output channel is not used
        self.output_channel = np.random.randint(1, 1001)
        self.num_layers = 5
        self.arch_params = None

        print(self.block_type, self.input_channel, self.output_channel, self.num_layers)

        # y
        self.latency = None

    def serialize_x(self):
        return np.append(np.array([
            self.block_type, self.input_channel,
            self.output_channel, self.num_layers,
        ]), reduce(
            lambda a, b: np.append(a, b), self.arch_params
        ))

    def process(self, load):
        """
        return: X, y, latency_list
        """
        # get latency and arch_params (randomly chosen in normal distribution)
        self.latency, self.arch_params, latency_list = LatencyEstimator(
            block_type=self.block_type,
            input_channel=self.input_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            dataset=load
        ).execute()

        return self.serialize_x(), self.latency, latency_list
