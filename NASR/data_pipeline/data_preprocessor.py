import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class PreProcessor:
    def __init__(self):
        # apply one-hot encoding to dataset
        self.dataset = pd.get_dummies(pd.read_csv("../data0520"), columns=["b_type"])

    @staticmethod
    def split_xy(df: pd.DataFrame):
        # return df[["b_type_0", "b_type_1", "in_ch"]], df["latency"]
        return df.drop(columns=["latency"]), df["latency"]

    def build_dataset(self):
        """
        load dataset and split dataset
        :return: train Xy, test Xy
        """
        # preprocessing
        self.dataset = self.preprocess()
        # split
        train, test = train_test_split(self.dataset, stratify=self.dataset["in_ch"])
        train_x, train_y = self.split_xy(train)
        test_x, test_y = self.split_xy(test)

        return train_x, train_y, test_x, test_y

    def preprocess(self):
        '''
        preprocessing!
        :return standard->minmax input channel, robust->minmax prob, minmax latency
        '''
        material = self.dataset
        prob = material.columns.difference(['b_type_0', 'b_type_1', 'latency', 'in_ch'])
        preprocess = make_column_transformer(
            (make_pipeline(StandardScaler(), MinMaxScaler()), ['in_ch']),
            (make_pipeline(RobustScaler(), MinMaxScaler()), prob),
            (MinMaxScaler(), ['latency']),
        )
        fitted = pd.DataFrame(preprocess.fit_transform(material), columns=material.columns[0:167])
        fitted = pd.concat([fitted, material.iloc[:, 167:169]], axis=1)

        return fitted
