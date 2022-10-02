# https://python.civic-apps.com/zipfile/
import io
import logging
import zipfile

logger = logging.getLogger(__name__)


class ZipFile:
    buffer: io.BytesIO = None
    file_data_dict: dict = {}

    def main(self, buffer=None):
        try:
            logger.info(f'{self.file_data_dict.keys()=}')
            if 0 == len(self.file_data_dict.keys()):
                raise AttributeError

            # with zipfile.ZipFile('foo.zip', "w", zipfile.ZIP_DEFLATED) as zf:
            if buffer is None:
                with io.BytesIO as buffer:
                    self.buffer = buffer

            self.buffer = buffer
            self.write_zipfile()

        except Exception as e:
            logger.error(e)
            raise e
        else:
            return self.buffer

    def write_zipfile(self):
        try:
            if self.buffer is None:
                raise AttributeError
            with zipfile.ZipFile(self.buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename, data in self.file_data_dict.items():
                    zf.writestr(filename, data)
        except Exception as e:
            logger.error(e)
            raise e

    def add_data(self, filename: str, data):
        try:
            if not isinstance(filename, str):
                raise TypeError
            self.file_data_dict[filename] = data
        except Exception as e:
            raise e

    def open_zipfile(self):
        with zipfile.ZipFile('foo.zip', 'r') as zf:
            with zf.open('foo.csv') as bar4:
                print(bar4)
