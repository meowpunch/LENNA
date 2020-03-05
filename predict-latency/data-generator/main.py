from generate_data import generate_data
import os
import time
import sys

import torchvision
import torchvision.transforms as transforms
import torch

import datetime

def dataload():

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
    # now_date = datetime.datetime.now()
    # print("now date: ", now_date)

    file_name = 'training_data' # + str(now_date)[:10].replace("-", "")
    test_ld = dataload()

    print("main start pid: %s" % (os.getpid()))
    # childs = {'$pa':'first child', '$pb':'second child'}

    children = []
    for j in range(2):
        j += 4
        # fork 3 child process
        child_pid = os.fork()

        if child_pid == 0:

            # each child fork and execute new_process 5000 times
            for i in range(5000):
                new_pid = os.fork()

                if new_pid == 0:
                    print("child %s" % (os.getpid()))
                    file_name += '_' + str(j)
                    # file_name is 'training_data_final_(0||1||2 etc)
                    if os.path.isfile('training-data/' + file_name) is True:
                        f = open('training-data/' + file_name, "a")
                    else:
                        f = open('training-data/' + file_name, "w")

                    for i in range(20):
                        cfg, target = generate_data(test_ld)
                        # print("in main")
                        print(j, "child count:", i)
                        # time.sleep(1)
                        f.writelines(cfg)
                        f.write(', ')
                        f.write(target)
                        f.write('\n')

                    f.close()
                    sys.exit(0)

                else:
                    print("parent(%s) got new_pid:%s" % (os.getpid(), new_pid))

                pid, status = os.waitpid(new_pid, 0)
                print("wait returned, pid = %d, status = %d" % (pid, status))

        """
        else:
            print("parent(%s) got new_pid:%s" % (os.getpid(), child_pid))
            children.append(child_pid)
        """

    # parent waits 3 child process
    if os.getpid() is not 0:
        status = os.wait()
        print("\nIn parent process-")
        print("Terminated child's process id:", status[0])
        print("Signal number that killed the child process:", status[1])


    """
        for j in range(100):
            # make_bucket()
            if os.path.isfile(file_name) is True:
                f = open(file_name, "a")
            else:
                f = open(file_name, "w")
    
            for i in range(100):
                cfg, target = generate_data(test_ld)
                # print("in main")
                print("count: ", i)
                # print(valid)
                # print(cfg)
                # print(target)
                # f.writelines(cfg)
                # f.write(', ')
                # f.write(target)
                # f.write('\n')
    
            f.close()
            # generate_data()
    """
