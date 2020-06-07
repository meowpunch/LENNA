import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, \
    QuantileTransformer


class PreProcessor:
    def __init__(self, filename="../final_data"):
        # apply one-hot encoding to dataset
        self.dataset = pd.get_dummies(pd.read_csv(filename), columns=["b_type"])

    def process(self):
        return self.build_dataset()

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

    @staticmethod
    def save(x, filename):
        dump(x, "../{}".format(filename))

    def latency_preprocess(self, latency):
        robust, quantile = RobustScaler(), QuantileTransformer(n_quantiles=1000,
                                                               output_distribution='normal')
        out = quantile.fit_transform(robust.fit_transform(latency.values.reshape(-1, 1)))

        self.save(robust, "robust.pkl")
        self.save(quantile, "quantile.pkl")
        return out

    def preprocess(self):
        '''
        preprocessing!
        :return standard->minmax input channel, robust->minmax prob, minmax latency
        '''
        material = self.dataset

        prob = material.columns.difference(['b_type_0', 'b_type_1', 'latency', 'in_ch'], sort=False)
        in_ch_lin = make_pipeline(StandardScaler(), \
                                  QuantileTransformer(n_quantiles=100, output_distribution='normal'))
        prob_lin = make_pipeline(MaxAbsScaler(), \
                                 QuantileTransformer(n_quantiles=1000, output_distribution='normal'))

        # latency_lin = make_pipeline(RobustScaler(), \
        #                             QuantileTransformer(n_quantiles=1000, output_distribution='normal'))

        preprocess = make_column_transformer(
            (in_ch_lin, ['in_ch']),
            (prob_lin, prob),
            # (latency_lin, ['latency'])
        )

        # latency
        latency = pd.DataFrame(self.latency_preprocess(material["latency"]), columns=["latency"])

        # block type 전처리 안하니까.
        fitted = pd.DataFrame(preprocess.fit_transform(material), columns=material.columns[0:-3])
        fitted = pd.concat([material[["b_type_0", "b_type_1"]], fitted, latency], axis=1)

        print(fitted.head(3))
        return fitted[fitted.b_type_1 == 1].drop(columns=["b_type_0", "b_type_1"])

