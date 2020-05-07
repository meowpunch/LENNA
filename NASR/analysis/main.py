import pandas as pd
import time

from analysis.binary_gates import accumulate_latency
from analysis.ops_latency import OpsAnalyzer
from data_pipeline.latency_estimator import LatencyEstimator
from util.dataset import load_dataset
from matplotlib import pyplot as plt


def research_ops_latency():
    df_list = OpsAnalyzer(counts=1000, size=(256, 16, 224, 224)).process()


def research_binary_gates():
    le = LatencyEstimator(
        block_type=0, input_channel=512, num_layers=5, dataset=load_dataset(batch_size=64)
    )
    le.model.init_arch_params()

    cumulative_avg, cumulative_err, cumulative_latency = accumulate_latency(le, 1000, 10)

    # plt.figure()
    # pd.Series(cumulative_err).plot()
    # plt.show()

    plt.figure()
    pd.Series(cumulative_avg).plot()
    plt.show()

    time.sleep(10)

    plt.figure()
    pd.Series(cumulative_latency).plot()
    plt.show()


def main():
    research_binary_gates()
    # research_ops_latency()


if __name__ == '__main__':
    main()
