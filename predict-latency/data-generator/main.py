from generate_data import generate_data
import os

if __name__ == '__main__':
    # file_name = './training_data'
    #
    # # make_bucket()
    # if os.path.isfile(file_name) is True:
    #     f = open(file_name, "a")
    # else:
    #     f = open(file_name, "w")
    #
    # for i in range(100):
    #     cfg, target = generate_data()
    #     print(cfg)
    #     print(target)
    #     f.writelines(cfg)
    #     f.write(', ')
    #     f.write(target)
    #     f.write('\n')
    #
    # f.close()
    generate_data()
