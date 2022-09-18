import logging

try:
    from ..services.image_service import ImageService
    from ..services.csv_service import CsvService
except ImportError:
    from services.image_service import ImageService
    from services.csv_service import CsvService

logger = logging.getLogger(__name__)


class Sidebar:
    radio_value: str
    radio_value_list = [
        'image_service',
        'csv_service',
        'etc'
    ]

    def __init__(self, st):
        self.st = st
        self.main()

    def main(self):
        self.st.sidebar.markdown('check sidebar')
        self._check_radio()

    def _check_radio(self):
        radio_value = self.st.sidebar.radio('check_radio', self.radio_value_list)

        if radio_value:
            self.radio_value = radio_value

    def image_service(self):
        uploaded_file = self.st.file_uploader('Choose a image file.', type=['jpeg', 'png'])
        try:
            if uploaded_file is not None:
                image_service = ImageService(fp=uploaded_file)
                self.st.image(image_service.image, caption='upload image', use_column_width=True)
        except Exception as e:
            logger.error(f'ERROR: {uploaded_file=}')

    def csv_service(self):
        uploaded_files = self.st.file_uploader("Choose a CSV file", type='csv', accept_multiple_files=False)
        try:
            if uploaded_files is not None:
                csv_service = CsvService(filepath_or_buffer=uploaded_files)

                self.st.markdown('- show upload csv file')
                self.st.dataframe(csv_service.df)

                self.st.markdown('- diff_column')
                self.st.dataframe(csv_service.calc_diff())

        except Exception as e:
            logger.error(f'ERROR: {uploaded_files=}')

    def etc_service(self):
        self.st.markdown("""
        |Page|Functions|
        |---|---|
        |image_service|upload|
        ||view|
        |csv_service|upload|
        ||view|
        |etc|*this page*|
        """)
