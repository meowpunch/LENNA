from Recasting_ver.core import DataGenerator


def main():
    """
        0: SuperDartsRecastingNet
        1: LennaNet
    :return:
    """
    dg = DataGenerator(mode=1)
    dg.execute()


if __name__ == '__main__':
    main()
