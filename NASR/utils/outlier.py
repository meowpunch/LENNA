def cut_outlier(df, min_border=None, max_border=None):
    if min_border is None and max_border is None:
        raise ValueError

    if min_border is None:
        return df[df.apply(lambda x: x < x.quantile(max_border), axis=0)]
    elif max_border is None:
        return df[df.apply(lambda x: x.quantile(min_border) < x, axis=0)]
    else:
        return df[df.apply(lambda x: (x.quantile(min_border) < x) and (x < x.quantile(max_border)), axis=0)]
