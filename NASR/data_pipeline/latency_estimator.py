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
        TODO: DataGenerator for predicting the latency of cell

    """

    def __init__(self, block_type, input_channel, output_channel, num_layers):
        self.logger = init_logger()

        # dataset
        self.train_loader = None
        self.test_loader = None

        self.model = LennaNet(self, normal_ops=normal_ops, reduction_ops=reduction_ops,
                              block_type=block_type, num_layers=num_layers,
                              input_channel=input_channel, output_channel=output_channel,
                              n_classes=10)  # for cifar10
        # avg for n times
        self.latency = None

        # move model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            cudnn.benchmark = True
        else:
            raise ValueError
        return

    def execute(self):
        return self.process()

    def process(self):
        """
        return: latency of one block
        """
        # load cifar10 dataset
        self.load_dataset()

        # init architecture parameters
        self.model.init_arch_params()

        # estimate latency of blocks
        self.expect_latency()
        return self.latency

    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='../Recasting_ver/data', train=True,
                                                 download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                                        shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(root='../Recasting_ver/data', train=False,
                                                download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                                       shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def expect_latency(self):
        with torch.no_grad():
            count = 1
            l_sum = 0
            for data in self.test_loader:
                if count > 2:
                    break

                images, labels = data

                # open the binary gate
                self.model.reset_binary_gates()
                # self.model.unused_modules_off()

                with torchprof.Profile(self.model, use_cuda=True) as prof:
                    outputs = self.model(images)

                # get latency
                latency = sum(get_time(prof, target="blocks", show_events=False))
                l_sum += latency
                self.logger.info("{n} - latency: {latency}, avg: {avg}".format(
                    n=count, latency=latency, avg=latency/count
                ))

                count += 1
            self.latency = latency/count


