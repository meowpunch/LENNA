import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from model.parent import BaseModel
from util.logger import init_logger
from util.visualize import draw_hist


class MLPRegressorModel(BaseModel):
    """
        MLPRegressor
    """

    def __init__(self, bucket_name, x_train, y_train, params=None):
        super().__init__(bucket_name, x_train, y_train, params, MLPRegressor)


class MLPRegressorSearcher(GridSearchCV):
    """
        for research
    """

    def __init__(
            self, x_train, y_train, bucket_name,
            grid_params=None, score=mean_squared_error
    ):
        if grid_params is None:
            raise ValueError("grid params are needed")

        self.x_train = x_train
        self.y_train = y_train
        self.scorer = score

        self.error = None  # pd.Series
        self.metric = None

        # s3
        self.s3_manager = None

        # logger
        self.logger = init_logger()

        super().__init__(
            estimator=MLPRegressor(learning_rate='adaptive'),
            param_grid=grid_params,
            scoring='neg_mean_absolute_error',
        )

    def fit(self, X=None, y=None, groups=None, **fit_params):
        super().fit(X=self.x_train, y=self.y_train)

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
        self.save_params(key="mlp_best_params.pkl".format(prefix=prefix))
        self.save_metric(key="mlp_metric.pkl".format(prefix=prefix))
        self.save_error_distribution(prefix=prefix)
        # self.save_model(key="{prefix}/mlp/model.pkl".format(prefix=prefix))

    def save_params(self, key):
        self.logger.info("tuned params: {params}".format(params=self.best_params_))
        dump(self.best_params_, key)

    def save_metric(self, key):
        self.logger.info("metric is {metric}".format(metric=self.metric))
        # self.s3_manager.save_dump(x=self.metric, key=key)

    def save_model(self, key):
        dump(self.best_estimator_, key)
        # save best elastic net
        # self.s3_manager.save_dump(self.best_estimator_, key=key)

    def save_error_distribution(self, prefix):
        draw_hist(self.error)
        # self.s3_manager.save_plt_to_png(
        #     key="{prefix}/image/error_distribution.png".format(prefix=prefix))
