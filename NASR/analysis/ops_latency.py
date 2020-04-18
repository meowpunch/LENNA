import functools

import pandas as pd
import torch
import torchprof

from model.test_model import MyModel1, MyModel2, MyModel3, Parallel1, Parallel2, Reduction
from util.latency import get_time
from util.latency import get_time
from util.logger import init_logger

borders = '-' * 30


class OpsAnalyzer:
    def __init__(self, counts=1000, size=(32, 1, 128, 128)):
        self.logger = init_logger()
        self.counts = counts
        # batch_size(32 or 64) X depth X width X height
        self.X = (torch.rand(size=size).uniform_() > 0.8).float()
        self.models = [MyModel1()]  # [MyModel1(), MyModel2(), MyModel3()]  # [MyModel4(), Parallel1(), Parallel2()]

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

        m_list = list(model.modules.keys())
        rows = []
        rows0 = []
        for i in range(self.counts):
            with torchprof.Profile(model, use_cuda=True) as prof:
                model(self.X)
            rows0.append(sum(get_time(prof=prof, target=model.__class__.__name__)))
            # print(prof.display(show_events=False))
            # self cpu time total of model [torchprof]

            """
                # latency per ops [torchprof]
                def get_latency(target, profiler):
                    return get_time(prof=profiler, target=target)[0]
    
                rows0.append(
                    list(map(
                        functools.partial(get_latency, profiler=prof), m_list
                    )))
            """

            with torch.autograd.profiler.profile() as prof:
                model(self.X)
            rows.append(prof.self_cpu_time_total)

            # model.latency_list = []
            model(self.X)
            # if i % 200 is 0:
            #     self.logger.info("estimate {i} times".format(i=i))

        self.logger.info(model.size_list)

        'torch.autograd.Profile'
        m_df = pd.DataFrame(rows, columns=["autograd"])
        'torchprof'
        m0_df = pd.DataFrame(rows0, columns=["torchprof"])
        'time'
        m1_df = pd.DataFrame(
            data=model.latency_list,
            columns=list(map(lambda x: x + "_time", m_list)) + ["total_time"]
        )
        # pd.concat([m0_df, m1_df], axis=1)
        # pd.concat([m_df, m1_df], axis=1)
        # pd.concat([m1_df, m_df, m0_df], axis=1)
        return pd.concat([m1_df, m_df, m0_df], axis=1)


def main():
    df_list = OpsAnalyzer(counts=10, size=(32, 1, 32, 32)).process()
    print(df_list)


if __name__ == '__main__':
    main()
