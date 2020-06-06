import datetime

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from model.elastic_net import ElasticNetModel, ElasticNetSearcher
from model.mlp_regressor import MLPRegressorModel, MLPRegressorSearcher
from util.logger import init_logger
from data_pipeline.data_preprocessor import PreProcessor

border = '-' * 50


class LatencyPredictModelPipeline:

    def __init__(self, dataset_name):
        self.logger = init_logger()
        self.date = datetime.datetime.now().strftime("%m%Y")

        self.dataset = PreProcessor(filename=dataset_name).process()  # self.build_dataset(filename=dataset_name)

    @staticmethod
    def split_xy(df: pd.DataFrame):
        # return df[["b_type_0", "b_type_1", "in_ch"]], df["latency"]
        return df.drop(columns=["latency"]), df["latency"]

    @staticmethod
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    def build_dataset(self, filename="../data0520.csv"):
        """
            load dataset and split dataset
        :return: train Xy, test Xy
        """
        # build dataset
        dataset = self.clean_dataset(pd.get_dummies(pd.read_csv(filename), columns=["b_type"]))

        # split
        # train, test = train_test_split(dataset[dataset.in_ch == 32].drop(columns=["in_ch"]))
        train, test = train_test_split(dataset, stratify=dataset["in_ch"], shuffle=True)
        train_x, train_y = self.split_xy(train)
        test_x, test_y = self.split_xy(test)

        return train_x, train_y, test_x, test_y

    def section(self, p_type, m_type, param):
        self.logger.info("{b} {p}_{m} {b}".format(b=border, p=p_type, m=m_type))
        if p_type is "tuned":
            return self.tuned_process(
                dataset=self.dataset,  # self.build_dataset()  # PreProcessor().process()
                param=param,
                m_type=m_type
            )
        elif p_type is "search":
            return self.search_process(
                dataset=self.dataset,  # self.build_dataset(),  # PreProcessor().process(),
                grid_params=param,
                m_type=m_type
            )
        else:
            raise NotImplementedError

    def process(self, process_type: str, model_type: str, param=None):
        try:
            return self.section(p_type=process_type, m_type=model_type, param=param)
        except NotImplementedError:
            self.logger.critical(
                "'{p_type}' is not supported. choose one of ['search','tuned']".format(p_type=process_type),
                exc_info=True)
            return 1
        except Exception as e:
            # TODO: consider that it can repeat to save one more time
            self.logger.critical(e, exc_info=True)
            return 1
        # return 0


    def inverse_latency(self, X):
        robust, quantile = load("robust.pkl"), load("quantile.pkl")
        self.logger.info("robust_param: {}".format(robust.center_))
        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        else:
            X = X.reshape(-1, 1)

        return robust.inverse_transform(quantile.inverse_transform(X)).reshape(-1)

    def search_process(self, dataset, m_type, grid_params):
        """
            ElasticNetSearcher for research
        :param dataset: merged 3 dataset (raw material price, terrestrial weather, marine weather)
        :param bucket_name: s3 bucket name
        :param term: term of researched dataset
        :param grid_params: grid for searching best parameters
        :return: metric (customized rmse)
        """
        train_x, train_y, test_x, test_y = dataset

        zoo = {
            "mlp": MLPRegressorSearcher,
            "enet": ElasticNetSearcher,
        }

        # hyperparameter tuning
        searcher = zoo[m_type](
            x_train=train_x, y_train=train_y, bucket_name=None,
            score=mean_absolute_error, grid_params=grid_params
        )
        searcher.fit(train_x, train_y)

        # predict & metric
        pred_y = searcher.predict(X=test_x)
        r_test, r_pred = self.inverse_latency(test_y), self.inverse_latency(pred_y)
        metric = searcher.estimate_metric(y_true=r_test, y_pred=r_pred)
        # metric = searcher.estimate_metric(y_true=test_y, y_pred=pred_y)

        # save
        # TODO self.now -> date set term, e.g. 010420 - 120420
        searcher.save(prefix="".format(date=self.date))
        # searcher.save_params(key="food_material_price_predict_model/research/tuned_params.pkl")
        return searcher

    def tuned_process(self, dataset, m_type, param):
        """
            tuned ElasticNet for production
        :param dataset: merged 3 dataset (raw material price, terrestrial weather, marine weather)
        :return: metric (customized rmse)
        """
        train_x, train_y, test_x, test_y = dataset

        zoo = {
            "mlp": MLPRegressorModel,
            "enet": ElasticNetModel,
        }
        # init model & fit
        model = zoo[m_type](
            bucket_name=None,
            x_train=train_x, y_train=train_y,
            params=param
        )
        model.fit()

        # adjust intercept for conservative prediction
        # model.model.intercept_ = model.model.intercept_ + 300

        # predict & metric
        pred_y = model.predict(X=test_x)
        r_test, r_pred = self.inverse_latency(test_y), self.inverse_latency(pred_y)
        metric = model.estimate_metric(scorer=mean_absolute_error, y_true=r_test, y_pred=r_pred)
        # metric = model.estimate_metric(scorer=mean_absolute_error, y_true=test_y, y_pred=pred_y)

        # save
        # TODO self.now -> date set term, e.g. 010420 - 120420
        model.save(prefix="".format(term=self.date))
        return model
