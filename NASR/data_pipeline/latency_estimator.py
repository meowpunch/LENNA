import os
from functools import reduce

import pandas as pd
import torch
import torchprof
from torch.backends import cudnn

from data_pipeline.lenna_net import LennaNet
from util.latency import get_time
from util.logger import init_logger

# constant
normal_ops = [
    '3x3_Conv', '5x5_Conv',
    '3x3_ConvDW', '5x5_ConvDW',
    '3x3_dConv', '5x5_dConv',
    '3x3_dConvDW', '5x5_dConvDW',
    '3x3_maxpool', '3x3_avgpool',
    'Zero',
    'Identity',
]
reduction_ops = [
    '3x3_Conv', '5x5_Conv',
    '3x3_ConvDW', '5x5_ConvDW',
    '3x3_dConv', '5x5_dConv',
    '3x3_dConvDW', '5x5_dConvDW',
    '2x2_maxpool', '2x2_avgpool',
]


class LatencyEstimator:
    """
        This class will estimate latency and return latency & arch params
    """

    def __init__(self, block_type, input_channel, num_layers, dataset):
        self.logger = init_logger()

        # dataset
        self.test_loader = dataset

        self.model = LennaNet(self, normal_ops=normal_ops, reduction_ops=reduction_ops,
                              block_type=block_type, num_layers=num_layers,
                              input_channel=input_channel, n_classes=10)  # for cifar10

        # move model to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        if device == 'cuda':
            self.p_model = torch.nn.DataParallel(module=self.model)
            cudnn.benchmark = True

        self.model.eval()

    def execute(self):
        return self.process()

    def process(self):
        """
        return: latency of one block & arch params & latency list(for jupyter notebook)
        """
        # init architecture parameters
        self.model.init_arch_params()

        # estimate latency of blocks
        l_list, l_avg = self.expect_latency()

        return list(map(
            lambda param: torch.Tensor.cpu(param).detach().numpy(),
            self.model.architecture_parameters()
        )), l_avg, l_list

    def analyze_latency(self, n_binary=10):
        """
            # TODO: remove outlier per once sampled binary gates
        :return: average of latency
        """
        # df = pd.DataFrame(columns=range(n_binary))
        l_series = []
        for i in range(n_binary):
            self.model.reset_binary_gates()
            l_list, l_avg = self.expect_latency(n_iter=10)
            print(pd.Series(l_list))
            l_series.append(pd.Series(l_list))

        def make_df(x, y):
            return pd.concat([x, y], axis=1)

        df = reduce(make_df, l_series)
        return df

    def expect_latency(self, n_iter=100):
        """
        :return: list of latency and average of latency
        """
        latency_avg = None
        latency_list = []
        with torch.no_grad():
            count = 1
            l_sum = 0
            for data in self.test_loader:
                if count > n_iter:
                    break

                images, labels = data

                # open the binary gate
                # self.model.reset_binary_gates()
                # self.model.unused_modules_off()

                with torch.autograd.profiler.profile() as prof:
                    self.p_model(images)

                # with torchprof.Profile(self.p_model, use_cuda=True) as prof:
                #     self.p_model(images)
                #
                # # get latency
                # latency = sum(get_time(prof, target="blocks", show_events=False))
                # l_sum += latency
                # latency_list.append(latency)
                # self.logger.info("{pid} worker)  {n} - latency: {latency}, avg: {avg}".format(
                #     pid=os.getpid(), n=count, latency=latency, avg=l_sum / count
                # ))

                count += 1
            latency_avg = l_sum / (count - 1)

        return latency_list, latency_avg
