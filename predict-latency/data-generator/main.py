from generate_data import generate_data
import os

def dataload():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test -accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    return testloader


if __name__ == '__main__':
    file_name = './training_data'
    test_ld = dataload()

    for j in range(10):
        # make_bucket()
        if os.path.isfile(file_name) is True:
            f = open(file_name, "a")
        else:
            f = open(file_name, "w")

        for i in range(100):
            cfg, target, valid = generate_data(test_ld)
            # print("in main")
            # print(valid)
            # print(cfg)
            # print(target)
            if valid is 1:
                f.writelines(cfg)
                f.write(', ')
                f.write(target)
                f.write('\n')

        f.close()
        # generate_data()
