import sys

from data_pipeline.main import parallel


def main(argv):
    print(argv)
    print(len(argv))

    if len(argv) is 1:
        destination = "training_data/training_data"
    else:
        destination = argv[1]
    parallel(destination=destination, p_num=4)


if __name__ == '__main__':
    main(sys.argv)
