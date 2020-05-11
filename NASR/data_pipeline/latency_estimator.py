import time
from functools import reduce

import pandas as pd
import torch
import torchprof
from torch.backends import cudnn

from data_pipeline.lenna_net_super import LennaNet
from util.latency import get_time
from util.logger import init_logger

# constant

# TODO: omit zero for summation
normal_ops = [
    '3x3_Conv', '5x5_Conv',
    '3x3_ConvDW', '5x5_ConvDW',
    '3x3_dConv', '5x5_dConv',
    '3x3_dConvDW', '5x5_dConvDW',
    '3x3_maxpool', '3x3_avgpool',
    # 'Zero',
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

    def __init__(self, block_type, input_channel, num_layers, dataset, gpu_id=0, parallel=False):
        self.logger = init_logger()

        # dataset
        self.test_loader = dataset
        self.model = LennaNet(num_blocks=[1], num_layers=num_layers, normal_ops=normal_ops, reduction_ops=reduction_ops,
                              block_type=block_type, input_channel=input_channel, n_classes=10)  # for cifar10

        # allocate 4 processes to 4 gpu respectively
        device = 'cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cuda'
        self.logger.info("assign to {}".format(device))

        self.device = torch.device(device)
        self.model.to(self.device)
        if device == 'cuda:{}'.format(gpu_id):
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
        self.model.init_arch_params(init_type='uniform', init_ratio=100)

        arch_params_prob = list(map(
            lambda param: torch.Tensor.cpu(param).detach().numpy().tolist(),
            self.model.arch_params_prob()
        ))
        arch_params = list(map(
            lambda param: torch.Tensor.cpu(param).detach().numpy().tolist(),
            self.model.architecture_parameters()
        ))
        self.logger.info("arch params: {}".format(arch_params))
        self.logger.info("arch params prob: {}".format(arch_params_prob))

        # estimate latency of blocks
        # latency = self.various_latency()
        latency = self.estimate_latency(max_reset_times=10000)

        return arch_params_prob, latency

    def estimate_latency(self, threshold=1, max_reset_times=10000):
        """
            1. sum the 40% value of the measured latency every 50 resets of the binary gate.
            2. get avg cumulative latency(sum)
            3. ratio of error and avg
            4. if the ratio is less than 1 continuously 10 times, break the loop

            TODO: change the threshold according to b_type and in_ch because threshold 1% is strictly for small value

        :return: latency
        """
        lat_sum, hit_num, pre_avg, cur_avg = 0, 0, 0, 0
        for i in range(max_reset_times):
            self.model.reset_binary_gates()
            self.logger.info("**{} times reset binary gate**".format(i))

            # the 40% value
            latency_df = self.one_block_latency(n_iter=10)
            # self.logger.info("one block latency describe: {}".format(latency_df.describe()))
            new_lat = latency_df.quantile(q=0.4)
            self.logger.info("newly estimated latency: {}".format(new_lat))
            lat_sum = lat_sum + new_lat

            # average
            cur_avg = lat_sum / (i + 1)
            self.logger.info("cumulative_avg, pre_avg: {}, {}".format(cur_avg, pre_avg))

            # ratio
            ratio = abs(cur_avg - pre_avg) / cur_avg * 100
            self.logger.info("convergence ratio: {}".format(ratio))

            # break by threshold & hit num
            if ratio < threshold and i >= 2:
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
        :return: inner one block: pd.DataFrame
        """
        with torch.no_grad():
            count = 1
            for idx, data in enumerate(self.test_loader):
                if count > n_iter:
                    break

                images, labels = data
                # self.logger.info("outer shape: {}".format(images.shape))

                # infer
                with torchprof.Profile(self.model, use_cuda=True) as prof:
                    self.model(images.cuda(self.device))

                count += 1
                if count % 10 == 0:
                    self.logger.info("{} times estimation".format(count))
            latency_df = self.model.blocks[0].latency_df
        return latency_df[latency_df.columns[-1]].rename("one_block_latency")

    def outer_total_latency(self, n_iter=100):
        """
        :return: outer latency
        """
        latency_list = []
        with torch.no_grad():
            count = 1
            for idx, data in enumerate(self.test_loader):
                if count > n_iter:
                    break

                images, labels = data
                # self.logger.info("outer shape: {}".format(images.shape))

                # infer
                start = time.time()
                self.model(images.cuda(self.device))
                latency_list.append((time.time() - start) * 1000000)

                count += 1
                if count % 10 == 0:
                    self.logger.info("{} times estimation".format(count))

        return pd.Series(latency_list, name="outer_latency_latency")

    def various_latency(self, n_iter=70):
        """
            inner total, outer total, ops of one block, the block
        :return: list of latency and average of latency
        """
        latency_avg = None
        outside_total_time = []
        torchprof_block_time = []
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
                # start = time.time()
                # self.model(images.cuda(self.device))
                # outside_total_time.append((time.time() - start))

                # autograd
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                #     self.p_model(images)
                # self.logger.info("autograd: {}".format(prof.self_cpu_time_total))

                # torchprof
                with torchprof.Profile(self.model, use_cuda=True) as prof:
                    start = time.time()
                    self.model(images.cuda(self.device))
                    outside_total_time.append((time.time() - start))
                torchprof_time = sum(get_time(prof, target="blocks", show_events=False))
                # self.logger.info("time: {}".format(self.model.latency_list))
                # self.logger.info("\n{}".format(self.model.blocks[0].latency_df))
                # self.logger.info("torchprof: {}".format(torchprof_time))
                torchprof_block_time.append(torchprof_time)

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
            torchprof_df = pd.DataFrame(data=torchprof_block_time, columns=["torchprof_block"])
            outside_df = pd.DataFrame(data=self.model.unit_transform(outside_total_time), columns=["outside_total"])
            combined_df = pd.concat([self.model.latency_df.rename(columns={0: "inside_total"}), outside_df,
                                     self.model.blocks[0].latency_df, torchprof_df],
                                    axis=1)  # .rename(columns={0: "inside_total", 1: "total"})
            from util.outlier import cut_outlier
            cut_df = cut_outlier(combined_df, min_border=0.25, max_border=0.75)
            self.logger.info("\n{}".format(combined_df))
            self.logger.info("\ntime: \n{} \nafter cut oulier: \n{}".format(
                combined_df.describe(),
                cut_df.describe()
            ))

        return combined_df, cut_df
