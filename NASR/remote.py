import sys

from data_pipeline.main import parallel, single


def main(argv):
    print(argv)
    print(len(argv))

    if len(argv) is 1:
        destination = "training_data/training_data"
    else:
        destination = argv[1]

    # total rows: outer * inner * p_num
    # parallel(destination=destination, outer_loop=250, inner_loop=10, p_num=4)
    single(destination=destination)


if __name__ == '__main__':
    main(sys.argv)
