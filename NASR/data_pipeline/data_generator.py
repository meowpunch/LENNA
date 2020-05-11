from functools import reduce

import numpy as np
import pandas as pd

from data_pipeline.latency_estimator import LatencyEstimator
from util.logger import init_logger


class DataGenerator:
    def __init__(self, sub_pid=0, g_type="random"):
        """
            block type: 0 -> reduction , 1-> normal
            input_channel: 1~1000
            num_layers: fix 5 or 6
        """
        self.logger = init_logger()
        self.sub_pid = sub_pid

        # X
        np.random.seed()
        self.block_type = np.random.randint(0, 2)
        self.input_channel = np.random.randint(1, 1025)
        self.num_layers = 5
        self.arch_params = None

        self.logger.info("init b_type, in_ch: {}, {} ".format(
            self.block_type, self.input_channel
        ))

        # y
        self.latency = None

    @property
    def serialize_x(self):
        if self.arch_params is None:
            raise NotImplementedError
        else:
            return np.append(np.array([
                self.block_type, self.input_channel,  # self.num_layers,
            ]), reduce(
                lambda a, b: np.append(a, b), self.arch_params
            ))

    def process(self, load, num_rows=100):
        """
        return: X, y, latency_list
        """
        # get latency and arch_params (randomly chosen in normal distribution)
        le = LatencyEstimator(
            block_type=self.block_type,
            input_channel=self.input_channel,
            num_layers=self.num_layers,
            gpu_id=self.sub_pid,
            dataset=load
        )

        def one_row(x):
            self.arch_params, self.latency = le.execute()
            return np.append(self.serialize_x, self.latency)

        df = pd.DataFrame(map(one_row, range(num_rows)))
        self.logger.info("show 5 rows\n {}".format(df.head(5)))

        return df
