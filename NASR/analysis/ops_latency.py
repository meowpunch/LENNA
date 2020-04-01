import numpy as np
import pandas as pd
import torch
import torchprof
from torch.autograd import Variable
import functools

from model.test_model import MyModel1, MyModel2, MyModel3, Parallel, Reduction
from utils.latency import get_time, get_df_times


class OpsAnalyzer:
    def __init__(self, counts=1000, size=20):
        self.counts = counts
        self.X = (torch.rand(1, 1, size, size).uniform_() > 0.8).float()
        self.models = [MyModel1(), MyModel2(), MyModel3(), Parallel()]  # Reduction()]

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
        m_list = model.choices.keys()
        rows = []
        for i in range(self.counts):
            with torchprof.Profile(model, use_cuda=True) as prof:
                model(self.X)
            print(prof.display(show_events=False))
            print(prof.display(show_events=True))

            def get_latency(target, profiler):
                return get_time(prof=profiler, target=target)[0]

            rows.append(
                list(map(
                    functools.partial(get_latency, profiler=prof), m_list
                )))
        return pd.DataFrame(rows, columns=m_list)


def main():
    df_list = OpsAnalyzer(counts=10).process()
    print(df_list)


if __name__ == '__main__':
    main()
