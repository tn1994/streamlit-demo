import logging
import datetime
from datetime import timedelta

try:
    from ..services.image_service import ImageService
    from ..services.csv_service import CsvService
    from ..services.stock_service import StockService
    from ..services.stock_service import color_survived
except ImportError:
    from services.image_service import ImageService
    from services.csv_service import CsvService
    from services.stock_service import StockService
    from services.stock_service import color_survived

logger = logging.getLogger(__name__)


class Sidebar:
    radio_value: str
    service_list = [
        'image_service',
        'csv_service',
        'stock_service',
        'etc'
    ]

    def __init__(self, st):
        self.st = st
        self.main()

    def main(self):
        option = self.st.sidebar.selectbox(
            'sub page:',
            Sidebar.service_list
        )

        match option:
            case 'image_service':
                self.st.title('image service')
                self.image_service()
            case 'csv_service':
                self.st.title('csv service')
                self.csv_service()
            case 'stock_service':
                self.st.title('storck service')
                self.stock_service()
            case 'etc':
                self.etc_service()
            case _:
                pass

    def _check_radio(self):
        radio_value = self.st.sidebar.radio('check_radio', self.service_list)
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

    def stock_service(self):
        stock_service = StockService()

        tab1, tab2, tab3 = self.st.tabs(['Check Signal', 'Check Per Ticker', 'Check Stock Value'])

        with tab1:  # check_signal
            self.st.info(f'Ticker: {", ".join(stock_service.ticker_list)}')
            with self.st.form('check_signal_form'):

                c1, c2, c3, c4 = self.st.columns([0.25, 0.25, 0.25, 0.25])

                with c1:
                    start_date = self.st.date_input('Start date', datetime.date(2020, 1, 1))
                with c2:
                    end_date = self.st.date_input('End date')
                with c3:
                    how_date = self.st.date_input('How long in the past do you check signals?',
                                                  datetime.date.today() - timedelta(7))

                if start_date > end_date:
                    self.st.error('Please start_date before end_date.')
                    is_check_signal_start_disabled: bool = True
                elif not (start_date < how_date < end_date):
                    self.st.error('Please start_date before how_date. '
                                  'And end_date after how_date.')
                    is_check_signal_start_disabled: bool = True
                else:
                    is_check_signal_start_disabled: bool = False

                with c4:
                    self.st.markdown('Get Signal Result')
                    submitted = self.st.form_submit_button(label='GET')

                if not is_check_signal_start_disabled and submitted:
                    with self.st.spinner('Wait for it...'):
                        stock_service.signal_check_main(start_date=start_date, end_date=end_date, how_date=how_date)
                    self.st.success('Success')
                    self.st.info(f'{start_date=}, {end_date=}, {how_date=}')
                    self.st.table(stock_service.get_result_signal_df())

        with tab2:  # check_per_ticker
            with self.st.form(key='check_per_ticker_form'):
                option = self.st.selectbox(
                    label='Ticker',
                    options=stock_service.ticker_list
                )
                submitted = self.st.form_submit_button(label='GET')

            if option is not None and submitted:
                with self.st.spinner('Wait for it...'):
                    stock_service.main(ticker=option)
                self.st.success('Success')
                self.st.info(f'ticker: {option}')
                with self.st.container():
                    self.st.markdown('## Close Value')
                    self.st.pyplot(stock_service.fig)
                with self.st.container():
                    self.st.markdown('## RCI')
                    self.st.pyplot(stock_service.rci_fig)
                with self.st.container():
                    self.st.markdown('## RSI')
                    self.st.pyplot(stock_service.rsi_fig)
                with self.st.container():
                    self.st.markdown('## MACD')
                    self.st.pyplot(stock_service.macd_fig)

        with tab3:  # check_stock_value
            is_view_df: bool = False

            with self.st.form(key='check_stock_value_form'):

                c1, c2, c3, c4 = self.st.columns([0.25, 0.25, 0.25, 0.25])

                with c1:
                    option = self.st.selectbox(
                        label='Ticker',
                        options=stock_service.ticker_list
                    )
                with c2:
                    start_date = self.st.date_input('Start date', datetime.date(2020, 1, 1))
                with c3:
                    end_date = self.st.date_input('End date')

                    if start_date > end_date:
                        self.st.error('Please start_date before end_date.')
                        is_check_stock_value_start_disabled: bool = True
                    else:
                        is_check_stock_value_start_disabled: bool = False
                with c4:
                    self.st.markdown('Get Stock Value')
                    submitted = self.st.form_submit_button(label='GET')

                if option is not None and not is_check_stock_value_start_disabled and submitted:
                    with self.st.spinner('Wait for it...'):
                        df = stock_service.get_stock(code=option, start_date=start_date, end_date=end_date)
                    self.st.info(f'Ticker: {option}')
                    self.st.line_chart(df.drop(columns=['Volume']))
                    self.st.bar_chart(df['Volume'])
                    is_view_df = True

            with self.st.expander(label='Stock Value Table', expanded=True):
                if is_view_df:
                    self.st.table(df)

    def etc_service(self):
        self.st.markdown("""
        |Sub Page       |Functions        |
        |---------------|-----------------|
        |image_service  |upload           |
        |               |view             |
        |csv_service    |upload           |
        |               |view             |
        |stock_service  |Check Signal     |
        |               |CHeck Per Ticker |
        |               |CHeck Stock Value|
        |etc            |*this page*      |
        """)
