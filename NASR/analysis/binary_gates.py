import pandas as pd
from matplotlib import pyplot as plt


def latency_binary_gate(le, n_iter=50):
    # le.model.blocks[0].reset_latency_list()
    le.model.reset_binary_gates()
    # latency = le.outer_total_latency(n_iter=n_iter)
    latency = le.one_block_latency(n_iter=n_iter)
    le.logger.info("result of {num} times estimation: \n{lat}".format(num=n_iter, lat=latency.describe().to_frame()))
    le.model.blocks[0].reset_latency_list()
    return latency


def accumulate_latency(le, max_reset_times=100, n_iter=50):
    latency_history = []

    avg_history = []
    err_history = []
    hit_count = 0
    for i in range(max_reset_times):
        le.logger.info("--------------- {} times reset binary gate ---------------".format(i))

        # accumulate latency of reset binary gate
        new_latency = latency_binary_gate(le, n_iter=n_iter).quantile(0.4)
        le.logger.info("new latency: {}".format(new_latency))
        latency_history.append(new_latency)

        # cumulative average
        if i is 0:
            avg_history.append(latency_history[0])
            avg_history.append(latency_history[0])
        else:
            avg_history.append((latency_history[-1] + avg_history[-1]*i) / (i + 1))

        # error between the latest cumulative avg and its previous
        err = avg_history[-1] - avg_history[-2]
        err_history.append(err)

        # show error history
        if i > 0:
            plt.figure()
            pd.Series(err_history[1:]).plot()
            plt.show()

        ratio = (abs(err_history[-1]) / avg_history[-1]) * 100
        le.logger.info("cumulative_avg, pre_avg: {}, {}".format(avg_history[-1], avg_history[-2]))
        le.logger.info("error: {}".format(err))
        le.logger.info("convergence ratio: {}".format(ratio))

        # break condition
        if ratio < 0.5 and i >= 10:
            hit_count = hit_count + 1
            le.logger.info("reset times, hit counts: {}, {}".format(i, hit_count))
            if hit_count is 10:
                break
        else:
            hit_count = 0
    return pd.Series(avg_history), pd.Series(err_history), pd.Series(latency_history)
