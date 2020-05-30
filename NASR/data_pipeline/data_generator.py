from functools import reduce

import numpy as np
import pandas as pd

from data_pipeline.latency_estimator import LatencyEstimator
from util.logger import init_logger


class DataGenerator:
    def __init__(self, sub_pid=0, g_type="random", b_type=None, in_ch=None):
        """
            block type: 0 -> reduction , 1-> normal
            input_channel: 1~1000
            num_layers: fix 5 or 6
        """
        self.logger = init_logger()
        self.sub_pid = sub_pid

        # # X
        # np.random.seed()
        # if b_type is None:
        #     self.block_type = np.random.randint(0, 2)
        # else:
        #     self.block_type = b_type
        # # 32 64 128 256
        # if in_ch is None:
        #     self.input_channel = np.random.randint(1, 1000)
        # else:
        #     self.input_channel = in_ch
        #
        # self.num_layers = 5
        # self.arch_params = None
        #
        # self.logger.info("init b_type, in_ch: {}, {} ".format(
        #     self.block_type, self.input_channel
        # ))
        #
        # # y
        # self.latency = None

    # @property
    # def serialize_x(self):
    #     if self.arch_params is None:
    #         raise NotImplementedError
    #     else:
    #         return np.append(np.array([
    #             self.block_type, self.input_channel,  # self.num_layers,
    #         ]), reduce(
    #             lambda a, b: np.append(a, b), self.arch_params
    #         ))

    def process(self, load, model, num_rows=10):
        """
        return: X, y, latency_list
        """
        # get latency and arch_params (randomly chosen in normal distribution)
        le = LatencyEstimator(
            model=model,
            gpu_id=self.sub_pid,
            dataset=load
        )

        df_0 = pd.DataFrame([[model.block_type, model.input_channel]], columns=["b_type", "in_ch"])

        def one_row(x):
            arch_params, latency = le.process(
                # init ratio represents the degree of training.
                init_ratio=np.random.choice([0.01, 0.05, 0.1, 1, 5, 10])
            )
            self.logger.info("{} rows ".format(x))
            return pd.concat([df_0, arch_params, pd.Series(latency, name="latency")], axis=1)

        df = reduce(lambda x, y: x.append(y), map(one_row, range(num_rows)))
        self.logger.info("show 5 rows\n {}".format(df.head(5)))

        return df
