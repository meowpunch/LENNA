import pandas as pd


def cut_outlier(x, min_border=0.1, max_border=0.9):
    """
    :param x: data
    :param min_border: min
    :param max_border: max
    :return: pd DataFrame or Series
    """
    if isinstance(x, list):
        x = pd.Series(x)

    if isinstance(x, pd.DataFrame):
        if min_border is None and max_border is None:
            raise ValueError

        if min_border is None:
            return x[x.apply(
                lambda y: (y < y.quantile(max_border)),
                axis=0)
            ]
        elif max_border is None:
            return x[x.apply(
                lambda y: y.quantile(min_border) < y,
                axis=0)
            ]
        else:
            return x[x.apply(
                lambda y: (y.quantile(min_border) < y) & (y < y.quantile(max_border)),
                axis=0)
            ]
    elif isinstance(x, pd.Series):
        if min_border is None and max_border is None:
            raise ValueError

        if min_border is None:
            return x[x.apply(lambda y: (y < x.quantile(max_border)))]
        elif max_border is None:
            return x[x.apply(lambda y: x.quantile(min_border) < y)]
        else:
            return x[x.apply(lambda y: (x.quantile(min_border) < y) & (y < x.quantile(max_border)))]
    else:
        raise NotImplementedError
