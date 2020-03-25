import numpy as np

from data_pipeline.latency_estimator import LatencyEstimator


class DataGenerator:
    def __init__(self):
        """
            block type: 0 -> reduction , 1-> normal
            input_channel: 1~1000
            output_channel: 1~1000
            num_layers: 1~10
        """
        # X
        np.random.seed()
        self.block_type = np.random.randint(0, 1)
        self.input_channel = np.random.randint(1, 512)
        self.output_channel = np.random.randint(1, 512)
        self.num_layers = np.random.randint(1, 5)

        # y
        self.latency = None

    def execute(self):
        return self.process()

    def process(self):
        """
        return: X, y
        """
        self.latency = LatencyEstimator(
            block_type=self.block_type,
            input_channel=self.input_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
        ).execute()

        return np.array([
            self.block_type, self.input_channel, self.output_channel, self.num_layers
        ]), self.latency



