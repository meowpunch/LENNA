from torchvision.transforms import transforms
import torchvision
import torch


def load_dataset(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = torchvision.datasets.CIFAR10(root='Recasting_ver/data', train=False,
                                            download=True, transform=transform)
    return torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                       shuffle=False, num_workers=2)
