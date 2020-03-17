from Recasting_ver.lenna_net import LennaNet
from Recasting_ver.cifar_arch_search import cifar_arch_search
import torch
import torchvision
import torchvision.transforms as transforms

import torchprof

normal_ops = [
    '3x3_Conv', '5x5_Conv',
    '3x3_ConvDW', '5x5_ConvDW',
    '3x3_dConv', '5x5_dConv',
    '3x3_dConvDW', '5x5_dConvDW',
    '3x3_maxpool', '3x3_avgpool',
    # 'Zero',
    # 'Identity',
]
reduction_ops = [
    '3x3_Conv', '5x5_Conv',
    '3x3_ConvDW', '5x5_ConvDW',
    '3x3_dConv', '5x5_dConv',
    '3x3_dConvDW', '5x5_dConvDW',
    '2x2_maxpool', '2x2_avgpool',
]
input_channel = 100
output_channel = 100
num_layers = 5
block_type = 0  # 0 -> reduction , 1-> normal  // should be one hot encoded


class DataGenerator:
    """
        TODO: DataGenerator for predicting the latency of cell

    """

    def __init__(self):
        self.model = LennaNet(self, num_layers=num_layers,
                              normal_ops=normal_ops, reduction_ops=reduction_ops, block_type=block_type,
                              input_channel=input_channel, output_channel=output_channel,
                              n_classes=10)  # for cifar10
        self.train_loader = None
        self.test_loader = None

        return

    def execute(self):
        return self.process()

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
            TODO: logic comes here
        """
        self.load_dataset()
        self.infer()
        return

    def train(self):
        pass

    def infer(self):
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                with torchprof.Profile(self.model, use_cuda=True) as prof:
                    outputs = self.model(images)
                print(prof.display(show_events=True))
                # print(prof.self_cpu_time_total)
        pass
