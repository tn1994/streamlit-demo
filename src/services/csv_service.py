import logging
import pandas as pd

logger = logging.getLogger(__name__)


class CsvService:

    def __init__(self, filepath_or_buffer):
        self.df = pd.read_csv(filepath_or_buffer=filepath_or_buffer)

    def calc_diff(self):
        diff_column = self.df.diff(axis=1)
        return diff_column
