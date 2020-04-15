import functools

import pandas as pd
import torch
import torchprof

from model.test_model import MyModel1, MyModel2, MyModel3, Parallel, Reduction
from util.latency import get_time
from util.logger import init_logger
import time

borders = '-' * 20


class OpsAnalyzer:
    def __init__(self, counts=1000, size=20):
        self.logger = init_logger()
        self.counts = counts
        # batch_size(32 or 64) X depth X width X height
        self.X = (torch.rand(1, 1, size, size).uniform_() > 0.8).float()
        self.models = [MyModel1(), MyModel2(), MyModel3()]  # , Parallel(), Reduction()]

    def execute(self):
        return self.process()

    def process(self):
        """
        return: list of pd DataFrame
        """
        return list(map(self.analyze, self.models))

    def analyze(self, model):
        """
        return: pd DataFrame
        """
        self.logger.info(borders + model.__class__.__name__ + borders)
        m_list = list(model.choices.keys())
        rows_total = []
        rows = []
        for i in range(self.counts):
            with torchprof.Profile(model, use_cuda=True) as prof:
                model(self.X)

            # print(prof.display(show_events=False))
            # print(prof.display(show_events=True))

            def get_latency(target, profiler):
                return get_time(prof=profiler, target=target)[0]

            rows.append(
                list(map(
                    functools.partial(get_latency, profiler=prof), m_list
                )))

        self.logger.info(model.size_list)

        # 'profiler'
        m0_df = pd.DataFrame(rows, columns=m_list)
        # 'time'
        m1_df = pd.DataFrame(
            data=model.latency_list,
            columns=list(map(lambda x: x + "_time", m_list)) + ["total_time"]
        )
        return pd.concat([m0_df, m1_df], axis=1)


def main():
    df_list = OpsAnalyzer(counts=1000, size=1024).process()
    print(df_list)


if __name__ == '__main__':
    main()
