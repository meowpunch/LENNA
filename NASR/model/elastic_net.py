import tempfile

import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from model.parent import BaseModel, BaseSearcher
from util.logger import init_logger
from util.visualize import draw_hist


class ElasticNetModel(BaseModel):
    """
        ElasticNet
    """

    def __init__(self, bucket_name: str, x_train, y_train, params=None):
        super().__init__(bucket_name, x_train, y_train, params, ElasticNet)

    @property
    def coef_df(self):
        """
        :return: pd DataFrame
        """
        return pd.Series(
            data=np.append(self.model.coef_, self.model.intercept_),
            index=self.x_train.columns.tolist() + ["intercept"],
        ).rename("beta").reset_index().rename(columns={"index": "column"})

    def save(self, prefix):
        """
            save tuned params, beta coef, metric, distribution, model
        :param prefix: dir
        """
        self.save_coef(key="{prefix}/beta.pkl".format(prefix=prefix))
        self.save_metric(key="{prefix}/metric.pkl".format(prefix=prefix))
        self.save_error_distribution(prefix=prefix)
        self.save_model(key="{prefix}/model.pkl".format(prefix=prefix))

    def save_coef(self, key):
        self.logger.info("beta_coef:\n{coef}".format(coef=self.coef_df))
        self.coef_df.to_csv("coef".format(key))


class ElasticNetSearcher(BaseSearcher):
    """
        for research
    """

    def __init__(self, x_train, y_train, bucket_name, grid_params=None, score=mean_squared_error):
        if grid_params is None:
            grid_params = {
                "max_iter": [1, 5, 10],
                "alpha": [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "l1_ratio": np.arange(0.0, 1.0, 0.1)
            }

        self.x_train = x_train
        self.y_train = y_train
        self.scorer = score

        self.error = None  # pd.Series
        self.metric = None

        # s3
        self.s3_manager = None

        # logger
        self.logger = init_logger()

        super().__init__(x_train, y_train, bucket_name, grid_params, score, ElasticNet)



    def fit(self, X=None, y=None, groups=None, **fit_params):
        super().fit(X=self.x_train, y=self.y_train)

    @property
    def coef_df(self):
        """
        :return: pd DataFrame
        """
        return pd.Series(
            data=np.append(self.best_estimator_.coef_, self.best_estimator_.intercept_),
            index=self.x_train.columns.tolist() + ["intercept"],
        ).rename("beta").reset_index().rename(columns={"index": "column"})

    def estimate_metric(self, y_true, y_pred):
        self.error = pd.Series(y_true - y_pred).rename("error")
        true = pd.Series(y_true)

        plt.figure()
        err_ratio = (abs(self.error) / true) * 100
        # err_ratio.plot()
        plt.scatter(x=true, y=err_ratio)
        plt.show()

        self.metric = self.scorer(y_true=y_true, y_pred=y_pred)
        return self.metric

    def save(self, prefix):
        """
            save tuned params, beta coef, metric, distribution, model
        :param prefix: dir
        """
        self.save_params(key="{prefix}/params.pkl".format(prefix=prefix))
        self.save_coef(key="{prefix}/beta.pkl".format(prefix=prefix))
        self.save_metric(key="{prefix}/metric.pkl".format(prefix=prefix))
        self.save_error_distribution(prefix=prefix)
        self.save_model(key="{prefix}/model.pkl".format(prefix=prefix))

    def save_params(self, key):
        self.logger.info("tuned params: {params}".format(params=self.best_params_))
        # self.s3_manager.save_dump(x=self.best_params_, key=key)

    def save_coef(self, key):
        self.logger.info("beta_coef:\n{coef}".format(coef=self.coef_df))
        self.coef_df.to_csv("coef".format(key))

    def save_metric(self, key):
        self.logger.info("metric is {metric}".format(metric=self.metric))
        # self.s3_manager.save_dump(x=self.metric, key=key)

    def save_model(self, key):
        # save best elastic net
        # self.s3_manager.save_dump(self.best_estimator_, key=key)
        pass

    def save_error_distribution(self, prefix):
        draw_hist(self.error)
        # self.s3_manager.save_plt_to_png(
        #     key="{prefix}/image/error_distribution.png".format(prefix=prefix))
