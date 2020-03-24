import Recasting_ver
from Recasting_ver.lenna_net import LennaNet
from Recasting_ver.cifar_arch_search_lenna import cifar_arch_search
import torch
import torchvision
import torchvision.transforms as transforms

import torchprof

# constant
from Recasting_ver.models.normal_nets.proxyless_nets import DartsRecastingBlock
from util.logger import init_logger

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
input_channel = 1000  # 1~1000
output_channel = 1000  # 1~1000
num_layers = 2  # 1~10
block_type = 0  # 0 -> reduction , 1-> normal  // should be one hot encoded


class DataGenerator:
    """
        TODO: DataGenerator for predicting the latency of cell

    """

    def __init__(self, mode=1):
        self.logger = init_logger()
        self.model = LennaNet(self, num_layers=num_layers,
                              normal_ops=normal_ops, reduction_ops=reduction_ops, block_type=block_type,
                              input_channel=input_channel, output_channel=output_channel,
                              n_classes=10)  # for cifar10
        self.mode = mode
        self.train_loader = None
        self.test_loader = None

        return

    def execute(self):
        if self.mode is 0:
            cifar_arch_search()
        elif self.mode is 1:
            self.process()

    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                                        shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                                       shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def process(self):
        """
            TODO:
                1. load dataset
                2. init arch params & print
                3. expect_latency inferring
        """
        self.load_dataset()

        self.model.init_arch_params()
        # print(list(self.model.architecture_parameters()))

        self.expect_latency()
        return

    def expect_latency(self):
        """
            TODO:
                1. open the binary gate by arch params


        :return:
        """
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data

                # open the binary gate
                self.model.reset_binary_gates()
                # self.model.unused_modules_off()

                # TODO: find torchprof variables (block cpu time)
                # with torchprof.Profile(self.model, use_cuda=True) as prof:
                outputs = self.model(images)

                self.logger.debug(self.model.get_latency())
                # self.logger.debug(prof.display(show_events=False))

                # self.model.unused_modules_back()

    def get_time(self, prof, show_events=False):
        """
        return: list of cpu total time (unit: ms)
        """
        traces = prof.traces
        trace_events = prof.trace_profile_events
        paths = prof.paths

        # self.logger.debug(traces)

        cpu_times = []

        for trace in traces:
            [path, leaf, module] = trace
            self.logger.debug(leaf)
            self.logger.debug(type(module))

            if isinstance(module, DartsRecastingBlock):
                # self.logger.debug(trace)
                events = [te for t_events in trace_events[path] for te in t_events]
                for depth, name in enumerate(path, 1):
                    if depth == len(path) and (
                            (paths is None and leaf) or (paths is not None and path in paths)
                    ):
                        if show_events:
                            for event in events:
                                cpu_times.append(event.cpu_time_total)
                        else:
                            cpu_times.append(sum([e.cpu_time_total for e in events]))
        return cpu_times

