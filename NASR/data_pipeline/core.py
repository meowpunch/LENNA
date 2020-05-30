import os
import sys

from data_pipeline.data_generator import DataGenerator
from data_pipeline.lenna_net_sub import LennaNet
from util.logger import init_logger


class DataPipeline:
    def __init__(self, idx, destination, lock):
        self.logger = init_logger()
        self.sub_pid = idx
        self.lock = lock

        # constant
        self.normal_ops = [
            '3x3_Conv', '5x5_Conv',
            '3x3_ConvDW', '5x5_ConvDW',
            '3x3_dConv', '5x5_dConv',
            '3x3_dConvDW', '5x5_dConvDW',
            '3x3_maxpool', '3x3_avgpool',
            # 'Zero',
            'Identity',
        ]
        self.reduction_ops = [
            '3x3_Conv', '5x5_Conv',
            '3x3_ConvDW', '5x5_ConvDW',
            '3x3_dConv', '5x5_dConv',
            '3x3_dConvDW', '5x5_dConvDW',
            '2x2_maxpool', '2x2_avgpool',
        ]

        self.destination = destination  # + str(idx)

        self.df = None

    def process(self, load, o_loop=250, shadow=True, i_loop=10, b_type=None, in_ch=None):
        """
        """
        i = 0
        model = LennaNet(num_blocks=[1], num_layers=5, normal_ops=self.normal_ops,
                         reduction_ops=self.reduction_ops, block_type=b_type,
                         input_channel=in_ch, n_classes=10)  # for cifar10
        if shadow:
            while i < o_loop:
                shadow = os.fork()
                latency_list = None
                if shadow == 0:
                    dg = DataGenerator(sub_pid=self.sub_pid)
                    self.df = dg.process(load=load, model=model, num_rows=i_loop)

                    self.save_to_csv()

                    sys.exit()
                else:
                    self.logger.info("%s worker got shadow %s" % (os.getpid(), shadow))

                pid, status = os.waitpid(shadow, 0)
                self.logger.info("wait returned, pid = %d, status = %d" % (pid, status))

                # TODO: increase 1 to i on shadow.
                i = i + 1
                self.logger.info("------------------------ {}*{} rows ------------------------".format(i, i_loop))
            return 0
        else:
            dg = DataGenerator()
            self.df = dg.process(load=load, num_rows=2)
            self.save_to_csv()
            return 0

    def save_to_csv(self):
        # lock
        if self.lock:
            self.lock.acquire()

        # save
        if os.path.isfile(self.destination) is True:
            self.df.to_csv(self.destination, mode='a', index=False, header=False)
        else:
            self.df.to_csv(self.destination, mode='w', index=False, header=True)
        self.logger.info("success to save data in '{dest}'".format(dest=self.destination))

        # unlock
        if self.lock:
            self.lock.release()
