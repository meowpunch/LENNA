import os

from Recasting_ver.cifar_arch_search_lenna import cifar_arch_search, cudnn
from data_pipeline.lenna_net import LennaNet
from utils.latency import get_time
from utils.logger import init_logger

import torch
import torchvision
import torchvision.transforms as transforms
import torchprof

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

    def __init__(self, block_type, input_channel, output_channel, num_layers, dataset):
        self.logger = init_logger()

        # dataset
        self.test_loader = dataset

        self.model = LennaNet(self, normal_ops=normal_ops, reduction_ops=reduction_ops,
                              block_type=block_type, num_layers=num_layers,
                              input_channel=input_channel, output_channel=output_channel,
                              n_classes=10)  # for cifar10
        # avg for n times
        self.latency = None

        # move model to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        if device == 'cuda':
            self.p_model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True

        self.model.eval()

    def execute(self):
        return self.process()

    def process(self):
        """
        return: latency of one block & arch params
        """
        # init architecture parameters
        self.model.init_arch_params()

        # estimate latency of blocks
        self.expect_latency()
        return self.latency, list(map(
            lambda param: torch.Tensor.cpu(param).detach().numpy(),
            self.model.architecture_parameters()
        ))

    def expect_latency(self):
        with torch.no_grad():
            count = 1
            l_sum = 0
            for data in self.test_loader:
                if count > 10000:
                    break

                images, labels = data

                # open the binary gate
                self.model.reset_binary_gates()
                # self.model.unused_modules_off()

                with torchprof.Profile(self.model, use_cuda=True) as prof:
                    outputs = self.p_model(images)

                # get latency
                latency = sum(get_time(prof, target="blocks", show_events=False))
                l_sum += latency
                self.logger.info("{pid} worker)  {n} - latency: {latency}, avg: {avg}".format(
                    pid=os.getpid(), n=count, latency=latency, avg=l_sum / count
                ))

                count += 1
            self.latency = latency / (count - 1)
