import datetime

import numpy as np

from analysis.food_material_price.research import split_xy, customized_rmse, search_process
from analysis.train_test_volume import train_test_timeseries
from model.elastic_net import ElasticNetModel
from util.build_dataset import build_master
from util.logging import init_logger
from util.s3_manager.manage import S3Manager

border = '-' * 50


class LatencyPredictModelPipeline:

    def __init__(self, bucket_name: str, logger_name: str, date: str):
        self.logger = init_logger()
        self.date = date
        # TODO: now -> term of dataset
        self.term = datetime.datetime.now().strftime("%m%Y")

        # s3
        self.bucket_name = bucket_name

    def build_dataset(self, pipe_data: bool):
        """
            TODO: standard date given from data_pipeline
        :param pipe_data: process data_pipeline or not (True/False)
        :return: split dataset
        """
        # TODO: it will be parameterized
        train_size, test_size = 5, 1

        # build dataset
        dataset = build_master(
            dataset="process_fmp", bucket_name=self.bucket_name,
            date=self.date, pipe_data=pipe_data
        )

        # set train, test dataset
        train, test = train_test_timeseries(
            df=dataset, train_size=train_size, test_size=test_size
        )
        self.logger.info("train/test: {train}/{test} day".format(train=train_size, test=test_size))
        train_x, train_y = split_xy(train)
        test_x, test_y = split_xy(test)
        return train_x, train_y, test_x, test_y

    def section(self, p_type, pipe_data: bool):
        self.logger.info("{b}{p_type}{b}".format(b=border, p_type=p_type))
        if p_type is "production":
            self.tuned_process(
                dataset=self.build_dataset(pipe_data=pipe_data)
            )
        elif p_type is "research":
            search_process(
                dataset=self.build_dataset(pipe_data=pipe_data),
                bucket_name=self.bucket_name, term=self.term,
                grid_params={
                    "max_iter": [1, 5, 10],
                    "alpha": [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    "l1_ratio": np.arange(0.0, 1.0, 0.1)
                }
            )
        else:
            raise NotImplementedError

    def process(self, process_type: str, pipe_data: bool):
        try:
            self.section(p_type=process_type, pipe_data=pipe_data)
        except NotImplementedError:
            self.logger.critical("'{p_type}' is not supported. choose one of ['research','production']".format(p_type=process_type), exc_info=True)
            return 1
        except Exception as e:
            # TODO: consider that it can repeat to save one more time
            self.logger.critical(e, exc_info=True)
            return 1
        return 0

    def tuned_process(self, dataset):
        """
            tuned ElasticNet for production
        :param dataset: merged 3 dataset (raw material price, terrestrial weather, marine weather)
        :return: metric (customized rmse)
        """
        train_x, train_y, test_x, test_y = dataset

        # init model & fit
        model = ElasticNetModel(
            bucket_name=self.bucket_name,
            x_train=train_x, y_train=train_y,
            params=S3Manager(bucket_name=self.bucket_name).load_dump(
                key="food_material_price_predict_model/research/tuned_params.pkl"
            )
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
        model.save(prefix="food_material_price_predict_model/{term}".format(term=self.term))
        return metric
