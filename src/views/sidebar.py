try:
    from ..services.image_service import ImageService
    from ..services.csv_service import CsvService
except ImportError:
    from services.image_service import ImageService
    from services.csv_service import CsvService


class Sidebar:
    radio_value: str

    def __init__(self, st):
        self.st = st
        self.main()

    def main(self):
        self.st.sidebar.markdown('check sidebar')
        self._check_radio()

    def _check_radio(self):
        self.radio_value_dict = {
            'image_service': self.st.write,
            'csv_service': self.st.write,
            'etc': self.st.write
        }
        radio_value = self.st.sidebar.radio('check_radio', self.radio_value_dict.keys())

        if radio_value:
            self.radio_value_dict[radio_value](f'checkd: {radio_value=}')
            self.radio_value = radio_value

    def image_service(self):
        uploaded_file = self.st.file_uploader('Choose a image file.')
        if uploaded_file is not None:
            image_service = ImageService(fp=uploaded_file)
            self.st.image(image_service.image, caption='upload image', use_column_width=True)

    def csv_service(self):
        uploaded_files = self.st.file_uploader("Choose a CSV file", accept_multiple_files=False)
        if uploaded_files is not None:
            csv_service = CsvService(filepath_or_buffer=uploaded_files)
            self.st.dataframe(csv_service.df)
