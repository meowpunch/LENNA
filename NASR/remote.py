import sys

from data_pipeline.main import parallel, single


def main(argv):
    """
        o_loop -> shadow
        i_loop -> num_rows
    """
    print(argv)
    print(len(argv))

    if len(argv) is 1:
        destination = "training_data/training_data"
    else:
        destination = argv[1]

    print("b_type: {}, in_ch: {}".format(argv[2], argv[3]))
    # total rows: outer * inner * p_num
    # parallel(destination=destination, outer_loop=250, inner_loop=10, p_num=4)
    single(destination=destination, o_loop=40, i_loop=50, b_type=int(argv[2]), in_ch=int(argv[3]))


if __name__ == '__main__':
    main(sys.argv)
