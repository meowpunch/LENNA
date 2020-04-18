def cut_outlier(df, min_border=None, max_border=None):
    if min_border is None and max_border is None:
        raise ValueError

    if min_border is None:
        return df.quantile([min_border, 1])
    elif max_border is None:
        return df.quantile([0, max_border])
    else:
        return df.quantile([min_border, max_border])
