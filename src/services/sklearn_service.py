import random
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score  # classification
from sklearn.metrics import r2_score  # regression
from sklearn.model_selection import train_test_split

# classification model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

SEED: int = 42

random.seed(SEED)  # random

np.random.seed(SEED)  # Numpy

logger = logging.getLogger(__name__)


class SklearnService:
    """
    ref: https://www.binarydevelop.com/article/scikitlearn-scikitlearn-69335
    """

    SEED = SEED

    X = None
    y = None

    predict_type_list = ['Classification', 'Regression']

    classification_model_dict = {
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'RandomForestClassifier': RandomForestClassifier
    }
    classification_model_list = classification_model_dict.keys()

    regression_model_dict = {
        'ExtraTreesRegressor': ExtraTreesRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor
    }
    regression_model_list = regression_model_dict.keys()

    model_name_dict = {
        'Classification': classification_model_list,
        'Regression': regression_model_list
    }

    def __init__(self, predict_type: str, model_name: str):
        try:
            if predict_type not in self.predict_type_list:
                raise ValueError
            if model_name not in self.classification_model_list and model_name not in self.regression_model_list:
                raise ValueError
            self.predict_type: str = predict_type
            self.model_name: str = model_name
            self.model = self._setup_model(predict_type=predict_type, model_name=model_name)
        except Exception as e:
            logger.error(e)
            raise e

    def _setup_model(self, predict_type: str, model_name: str):
        try:
            print(f'{predict_type=}, {model_name=}')

            if predict_type not in self.predict_type_list:
                raise ValueError

            match predict_type:
                case 'Classification':
                    model = self.classification_model_dict[model_name]()
                case 'Regression':
                    model = self.regression_model_dict[model_name]()
                case _:
                    raise ValueError

            print(f'{model=}')

            if model is None:
                raise ValueError
        except Exception as e:
            logger.error(e)
            raise e
        else:
            return model

    def main(self, df: pd.DataFrame, predict_column_name: str):
        try:
            if isinstance(df, pd.DataFrame):
                X = df.drop(predict_column_name, axis=1)
                y = df[predict_column_name]
            else:
                raise TypeError

            train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=.7, random_state=self.SEED,
                                                                shuffle=True)

            self.fit(train_x, train_y)
            logger.info(f'check fit: ')

            self.show_feature_importances()

            y_pred = self.predict(test_x)
            y_true = test_y

            match self.predict_type:
                case 'Classification':
                    logger.info(f'train result: {self.get_accuracy_score(train_y, self.predict(train_x))}')
                    self.result_score = self.get_accuracy_score(y_true=y_true, y_pred=y_pred)
                case 'Regression':
                    logger.info(f'train result: {self.get_r2_score(train_y, self.predict(train_x))}')
                    self.result_score = self.get_r2_score(y_true=y_true, y_pred=y_pred)
                case _:
                    raise ValueError

            self.submission_df: pd.DataFrame = pd.concat(
                [
                    test_x,
                    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
                ],
                axis=1
            )
        except Exception as e:
            logger.error(e)
            raise e
        else:
            return y_pred

    def fit(self, X, y):
        try:
            if self.model is None:
                raise AttributeError
            self.model.fit(X, y)
        except Exception as e:
            raise e

    def predict(self, X):
        try:
            if self.model is None:
                raise AttributeError
            y_pred = self.model.predict(X)
        except Exception as e:
            raise e
        else:
            return y_pred

    def show_feature_importances(self):
        try:
            if self.model is None:
                raise AttributeError
            logger.info(self.model.feature_importances_)
        except Exception as e:
            raise e

    def get_accuracy_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def get_r2_score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
