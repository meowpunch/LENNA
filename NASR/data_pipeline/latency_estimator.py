import time
from functools import reduce

import pandas as pd
import torch
from torch.backends import cudnn

from data_pipeline.lenna_net import LennaNet
from util.logger import init_logger

# constant

# TODO: omit zero for summation
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

    def __init__(self, block_type, input_channel, num_layers, dataset, parallel=False):
        self.logger = init_logger()

        # dataset
        self.test_loader = dataset

        self.model = LennaNet(self, normal_ops=normal_ops, reduction_ops=reduction_ops,
                              block_type=block_type, num_layers=num_layers,
                              input_channel=input_channel, n_classes=10)  # for cifar10

        # move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        if device == 'cuda':
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
        latency = self.estimate_latency(max_reset_times=1000)

        return list(map(
            lambda param: torch.Tensor.cpu(param).detach().numpy(),
            self.model.architecture_parameters()
        )), latency

    def estimate_latency(self, max_reset_times=10000):
        """
            1. sum the 40% value of the measured latency every 50 resets of the binary gate.
            2. get avg cumulative latency(sum)

        :return: average of latency
        """
        lat_sum, hit_num, pre_avg, cur_avg = 0, 0, 0, 0
        for i in range(max_reset_times):
            self.model.reset_binary_gates()

            # the 40% value
            cur_lat = self.outer_total_latency(n_iter=50).quantile(q=0.4)
            lat_sum = lat_sum + cur_lat

            # average
            cur_avg = lat_sum / (i + 1)

            # ratio
            ratio = abs(cur_avg - pre_avg) / cur_avg * 100

            self.logger.info("cumulative_avg, pre_avg: {}, {}".format(cur_avg, pre_avg))
            self.logger.info("convergence ratio: {}".format(ratio))
            if ratio < 1 and i >= 50:
                hit_num = hit_num + 1
                self.logger.info("reset times, hit counts: {}, {}".format(i, hit_num))
                if hit_num is 10:
                    self.logger.info("final latency: {}".format(cur_avg))
                    break
            else:
                hit_num = 0

            pre_avg = cur_avg

        return cur_avg

    def one_block_latency(self, n_iter=100):
        """
        :return: inner one block
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

                # infer
                self.model(images.cuda())

                count += 1
                if count % 10 == 0:
                    self.logger.info("{} times estimation".format(count))

            latency = pd.Series(data=self.model.blocks[0].latency_list[15], name="latency")

        return latency

    def outer_total_latency(self, n_iter=100):
        """
        :return: outer latency
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

                # infer
                start = time.time()
                self.model(images.cuda())
                latency_list.append((time.time() - start) * 1000000)

                count += 1
                if count % 10 == 0:
                    self.logger.info("{} times estimation".format(count))

        return pd.Series(latency_list, name="latency")

    def research_latency(self, n_iter=70):
        """
            inner total, outer total, ops of one block, the block
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
                # self.logger.info("outer shape: {}".format(images.shape))

                # open the binary gate
                # self.model.reset_binary_gates()
                # self.model.unused_modules_off()

                # time
                start = time.time()
                self.model(images.cuda())
                outside_total_time.append((time.time() - start))

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
                if count % 10 == 0:
                    self.logger.info("{} times estimation".format(count))
            # for block in self.model.blocks:
            #     self.logger.info("{}".format(block.latency_df))
            outside_df = pd.DataFrame(data=self.model.unit_transform(outside_total_time), columns=["outside_total"])
            combined_df = pd.concat([self.model.latency_df.rename(columns={0: "inside_total"}), outside_df,
                                     self.model.blocks[0].latency_df],
                                    axis=1)  # .rename(columns={0: "inside_total", 1: "total"})
            from util.outlier import cut_outlier
            cut_df = cut_outlier(combined_df, min_border=0.25, max_border=0.75)
            self.logger.info("\n{}".format(combined_df))
            self.logger.info("\ntime: \n{} \nafter cut oulier: \n{}".format(
                combined_df.describe(),
                cut_df.describe()
            ))

        return combined_df, cut_df
