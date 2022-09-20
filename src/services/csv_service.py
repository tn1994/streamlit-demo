import io
import random
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class CsvService:

    def __init__(self, filepath_or_buffer):
        try:
            self.df = pd.read_csv(filepath_or_buffer=filepath_or_buffer)
        except Exception as e:
            raise e

    def calc_diff(self):
        diff_column = self.df.diff(axis=1)
        return diff_column


def get_classification_data() -> pd.DataFrame:
    # 0: 快適では無い, 1: 快適である
    temperature_name = 'temperature'  # 気温
    humidity_name = 'humidity'  # 湿度
    temperature_list, humidity_list = generate_data()
    df = pd.DataFrame({
        temperature_name: temperature_list,
        humidity_name: humidity_list,
        'is_comfortable': 0
    })
    df.loc[(25 <= df[temperature_name]) & (df[temperature_name] <= 28) & (45 <= df[humidity_name]) & (
            df[humidity_name] <= 60), 'is_comfortable'] = 1
    return df


def get_classification_buffer_data() -> io.StringIO:
    return io.StringIO(get_classification_data().to_csv(index=False))


def get_regression_data() -> pd.DataFrame:
    df = pd.DataFrame({
        'a': [random.randint(0, 100) for _ in range(1000)],
        'b': [random.randint(0, 100) for _ in range(1000)],
        'c': [random.randint(0, 100) for _ in range(1000)],
        'd': [random.randint(0, 100) for _ in range(1000)],
        'e': [random.randint(0, 100) for _ in range(1000)]
    })
    return df


def get_regression_buffer_data():
    return io.StringIO(get_regression_data().to_csv(index=False))


def generate_data():
    base_list = [i for i in range(100)]
    temperature_list = []
    humidity_list = []

    for i in range(100):
        temperature_list.extend(base_list)
        humidity_list.extend([i] * 100)

    return temperature_list, humidity_list
