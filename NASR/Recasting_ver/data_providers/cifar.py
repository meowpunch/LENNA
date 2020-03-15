# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Recasting_ver.data_providers.base_provider import *


class CifarDataProvider(DataProvider):

    def __init__(self, save_path=None, n_classes=10, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        self.nc = n_classes
        train_transforms = self.build_train_transform()

        print(self._save_path)
        if self.n_classes == 10 :
            train_dataset = datasets.CIFAR10(root=self._save_path, train=True,
                                         download=True, transform=train_transforms)
        elif self.n_classes == 100:
            train_dataset = datasets.CIFAR100(root=self._save_path, train=True,
                                         download=True, transform=train_transforms)
        else :
            raise NotImplementedError

        self.valid = None
        
        test_transforms = transforms.Compose([transforms.ToTensor(), self.normalize])
        if self.n_classes == 10 :
            test_dataset = datasets.CIFAR10(root=self._save_path, train=False,
                                        download=True, transform=test_transforms)
        elif self.n_classes == 100:
            test_dataset = datasets.CIFAR100(root=self._save_path, train=False,
                                        download=True, transform=test_transforms)
        else :
            raise NotImplementedError

        self.train = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                 shuffle=True, num_workers=n_worker, pin_memory=True)
        self.test = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                                 shuffle=False, num_workers=n_worker, pin_memory=True)

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return self.nc

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/dataset/cifar'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download Cifar')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    def build_train_transform(self):
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,4),
            transforms.ToTensor(),
            self.normalize,
        ])
        return train_transforms

    @property
    def resize_value(self):
        return 32

    @property
    def image_size(self):
        return 32
