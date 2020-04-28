import os
import time
from functools import reduce

import pandas as pd
import torch
import torchprof
from torch.backends import cudnn

from data_pipeline.lenna_net import LennaNet
from util.latency import get_time
from util.logger import init_logger

# constant
from util.outlier import cut_outlier

# TODO: omit zero for summation
normal_ops = [
    '3x3_Conv', '5x5_Conv',
    '3x3_ConvDW', '5x5_ConvDW',
    '3x3_dConv', '5x5_dConv',
    '3x3_dConvDW', '5x5_dConvDW',
    '3x3_maxpool', '3x3_avgpool',
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

    def __init__(self, block_type, input_channel, num_layers, dataset, parallel=False):
        self.logger = init_logger()

        # dataset
        self.test_loader = dataset

        self.model = LennaNet(self, normal_ops=normal_ops, reduction_ops=reduction_ops,
                              block_type=block_type, num_layers=num_layers,
                              input_channel=input_channel, n_classes=10)  # for cifar10

        # move model to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        if device == 'cuda':
            self.model.cuda();
            # self.p_model = torch.nn.DataParallel(module=self.model)
            cudnn.benchmark = True

        self.model.eval()

    def execute(self):
        return self.process()

    def process(self):
        """
        return: latency of one block & arch params & latency list(for jupyter notebook)
        """
        # init architecture parameters by uniform distribution
        self.model.init_arch_params()

        # estimate latency of blocks
        # l_list, l_avg = self.research()
        latency = self.get_latency()

        return list(map(
            lambda param: torch.Tensor.cpu(param).detach().numpy(),
            self.model.architecture_parameters()
        )), latency

    def get_latency(self, reset_times=3):
        """
            # TODO: remove outlier per once sampled binary gates
        :return: average of latency
        """
        # df = pd.DataFrame(columns=range(n_binary))
        latency_by_binary_gates = []
        for i in range(reset_times):
            self.model.reset_binary_gates()
            latency_by_binary_gates.append(self.expect_latency(n_iter=1000))

        self.logger.info("latency by binary gates: {}".format(latency_by_binary_gates))

        return sum(latency_by_binary_gates) / len(latency_by_binary_gates)

    def expect_latency(self, n_iter=70):
        """
            for analysis
        :return: list of latency and average of latency
        """
        latency_list = []
        with torch.no_grad():
            count = 1
            l_sum = 0
            for idx, data in enumerate(self.test_loader):
                if count > n_iter:
                    break

                images, labels = data
                # self.logger.info("outer shape: {}".format(images.shape))

                # time
                start = time.time()
                self.model(images)
                latency_list.append((time.time() - start) * 1000000)  # sec to micro sec

                count += 1

            latency = pd.Series(data=latency_list, name="latency")
            filtered = cut_outlier(latency, min_border=0.25, max_border=0.75)

            if count%100 is 0:
                self.logger.info("{} times estimation".format(count))
            # describe make time more complex,
            # self.logger.info("\nlatency: \n{} \nafter filtering: \n{}".format(
            #     latency.describe(), filtered.describe()
            # ))
        return filtered.mean()

    def research_get_latency(self, reset_times=10):
        """
            # TODO: remove outlier per once sampled binary gates
        :return: average of latency
        """
        # df = pd.DataFrame(columns=range(n_binary))
        l_series = []
        for i in range(reset_times):
            self.model.reset_binary_gates()
            l_list, l_avg = self.expect_latency(n_iter=10)
            l_series.append(pd.Series(l_list))

        def make_df(x, y):
            return pd.concat([x, y], axis=1)

        df = reduce(make_df, l_series)
        return df

    def research_expect_latency(self, n_iter=70):
        """
            for analysis
        :return: list of latency and average of latency
        """
        latency_avg = None
        outside_total_time = []
        latency_list = []
        with torch.no_grad():
            count = 1
            l_sum = 0
            for idx, data in enumerate(self.test_loader):
                if count > n_iter:
                    break

                images, labels = data
                self.logger.info("outer shape: {}".format(images.shape))

                # open the binary gate
                # self.model.reset_binary_gates()
                # self.model.unused_modules_off()

                # time
                start = time.time()
                self.p_model(images)
                outside_total_time.append((time.time() - start) * 1000000)

                # autograd
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                #     self.p_model(images)
                # self.logger.info("autograd: {}".format(prof.self_cpu_time_total))

                # torchprof
                # with torchprof.Profile(self.p_model, use_cuda=True) as prof:
                #     self.p_model(images)
                # self.logger.info("time: {}".format(self.p_model.module.latency_list))
                # self.logger.info("torchprof: {}".format(sum(get_time(prof, target="blocks", show_events=False))))

                # get latency
                # latency = sum(get_time(prof, target="blocks", show_events=False))
                # l_sum += latency
                # latency_list.append(latency)
                # self.logger.info("{n} times - latency: {latency}, avg: {avg}".format(
                #     pid=os.getpid(), n=count, latency=latency, avg=l_sum / count
                # ))

                count += 1

            outside_df = pd.DataFrame(data=outside_total_time, columns=["outside_total"])
            combined_df = pd.concat([self.p_model.module.latency_df, outside_df], axis=1)
            from util.outlier import cut_outlier
            self.logger.info("time: \n{} \n{}".format(
                cut_outlier(combined_df[4:].rename(columns={0: "block", 1: "total"}),
                            min_border=0.25, max_border=0.75).describe(),
                combined_df.describe()
            ))
            latency_avg = l_sum / (count - 1)

        return latency_list, latency_avg
