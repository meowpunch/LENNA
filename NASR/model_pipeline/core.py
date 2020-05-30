import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from model.elastic_net import ElasticNetModel, ElasticNetSearcher
from util.logger import init_logger

border = '-' * 50


class LatencyPredictModelPipeline:

    def __init__(self):
        self.logger = init_logger()
        self.date = datetime.datetime.now().strftime("%m%Y")

    @staticmethod
    def split_xy(df: pd.DataFrame):
        # return df[["b_type_0", "b_type_1", "in_ch"]], df["latency"]
        return df.drop(columns=["latency"]), df["latency"]

    def build_dataset(self):
        """
            load dataset and split dataset
        :return: train Xy, test Xy
        """
        # build dataset
        dataset = pd.get_dummies(pd.read_csv("../data0520"), columns=["b_type"])

        # split
        train, test = train_test_split(dataset, stratify=dataset["in_ch"])
        train_x, train_y = self.split_xy(train)
        test_x, test_y = self.split_xy(test)

        return train_x, train_y, test_x, test_y

    def section(self, p_type):
        self.logger.info("{b}{p_type}{b}".format(b=border, p_type=p_type))
        if p_type is "tuned":
            self.tuned_process(
                dataset=self.build_dataset()
            )
        elif p_type is "search":
            self.search_process(
                dataset=self.build_dataset(), term=self.date,
                grid_params={
                    "max_iter": [1, 5, 10],
                    "alpha": [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    "l1_ratio": np.arange(0.0, 1.0, 0.1)
                }
            )
        else:
            raise NotImplementedError

    def process(self, process_type: str):
        try:
            self.section(p_type=process_type)
        except NotImplementedError:
            self.logger.critical(
                "'{p_type}' is not supported. choose one of ['search','tuned']".format(p_type=process_type),
                exc_info=True)
            return 1
        except Exception as e:
            # TODO: consider that it can repeat to save one more time
            self.logger.critical(e, exc_info=True)
            return 1
        return 0

    def search_process(self, dataset, term, grid_params):
        """
            ElasticNetSearcher for research
        :param dataset: merged 3 dataset (raw material price, terrestrial weather, marine weather)
        :param bucket_name: s3 bucket name
        :param term: term of researched dataset
        :param grid_params: grid for searching best parameters
        :return: metric (customized rmse)
        """
        train_x, train_y, test_x, test_y = dataset

        # hyperparameter tuning
        searcher = ElasticNetSearcher(
            x_train=train_x, y_train=train_y, bucket_name=None,
            score=mean_absolute_error, grid_params=grid_params
        )
        searcher.fit(train_x, train_y)

        # predict & metric
        pred_y = searcher.predict(X=test_x)
        # r_test, r_pred = inverse_price(test_y), inverse_price(pred_y)
        metric = searcher.estimate_metric(y_true=test_y, y_pred=pred_y)

        # save
        # TODO self.now -> date set term, e.g. 010420 - 120420
        searcher.save(prefix="../result/{date}".format(date=term))
        searcher.save_params(key="food_material_price_predict_model/research/tuned_params.pkl")
        return metric

    def tuned_process(self, dataset):
        """
            tuned ElasticNet for production
        :param dataset: merged 3 dataset (raw material price, terrestrial weather, marine weather)
        :return: metric (customized rmse)
        """
        train_x, train_y, test_x, test_y = dataset

        # init model & fit
        model = ElasticNetModel(
            x_train=train_x, y_train=train_y,
            params=None
        )
        model.fit()

        # adjust intercept for conservative prediction
        model.model.intercept_ = model.model.intercept_ + 300

        # predict & metric
        pred_y = model.predict(X=test_x)
        # r_test, r_pred = inverse_price(test_y), inverse_price(pred_y)
        metric = model.estimate_metric(scorer=customized_rmse, y=test_y, predictions=pred_y)

        # save
        # TODO self.now -> date set term, e.g. 010420 - 120420
        model.save(prefix="food_material_price_predict_model/{term}".format(term=self.date))
        return metric