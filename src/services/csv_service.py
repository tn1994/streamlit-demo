import pandas as pd


class CsvService:

    def __init__(self, filepath_or_buffer):
        self.df = pd.read_csv(filepath_or_buffer=filepath_or_buffer)
