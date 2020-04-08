import functools

import pandas as pd
import torch
import torchprof

from model.test_model import MyModel1, MyModel2, MyModel3, Parallel, Reduction
from util.latency import get_time
from util.logger import init_logger

borders = '-' * 20


class OpsAnalyzer:
    def __init__(self, counts=1000, size=20):
        self.logger = init_logger()
        self.counts = counts
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
        m_list = model.choices.keys()
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
        return pd.DataFrame(rows, columns=m_list)


def main():
    df_list = OpsAnalyzer(counts=10, size=128).process()
    print(df_list)


if __name__ == '__main__':
    main()
