import logging
import datetime
from datetime import timedelta

import streamlit as st

logger = logging.getLogger(__name__)

try:
    from ..services.image_service import ImageService
    from ..services.csv_service import CsvService
    from ..services.csv_service import get_classification_buffer_data
    from ..services.csv_service import get_regression_buffer_data
    from ..services.calc_service import CalcService
    from ..services.stock_service import StockService
    from ..services.notion_service import NotionService

    from ..services.stock_service import color_survived
    from ..services.sklearn_service import SklearnService
except ImportError:
    logger.info('check: ImportError')  # todo: fix import error
    from services.image_service import ImageService
    from services.csv_service import CsvService
    from services.csv_service import get_classification_buffer_data
    from services.csv_service import get_regression_buffer_data
    from services.calc_service import CalcService
    from services.stock_service import StockService

    from services.notion_service import NotionService

    from services.stock_service import color_survived
    from services.sklearn_service import SklearnService


class Sidebar:
    radio_value: str

    def __init__(self):
        self.service_dict = {
            'image_service': self.image_service,
            'csv_service': self.csv_service,
            'stock_service': self.stock_service,
            'calc_service': self.calc_service,
            'markdown_service': self.markdown_service,
            'notion_service': self.notion_service,
            'etc': self.etc_service
        }

    def main(self):
        option = st.sidebar.selectbox(
            'Sub Page',
            self.service_dict.keys()
        )
        select_service = self.service_dict[option]
        select_service()

    def _check_radio(self):
        radio_value = st.sidebar.radio('check_radio', self.service_dict.keys())
        if radio_value:
            self.radio_value = radio_value

    def image_service(self):
        st.title('Image service')
        uploaded_file = st.file_uploader('Choose a image file.', type=['jpeg', 'png'])
        try:
            if uploaded_file is not None:
                image_service = ImageService(fp=uploaded_file)
                st.image(image_service.image, caption='upload image', use_column_width=True)
        except Exception as e:
            logger.error(f'ERROR: {uploaded_file=}')

    def csv_service(self):
        st.title('CSV service')

        tab1, tab2 = st.tabs(['Use Your CSV File', 'Use Temp Data'])

        if 'is_use_tmp_classification_data' not in st.session_state and 'is_use_tmp_regression_data' not in st.session_state:
            st.session_state.is_use_tmp_classification_data = False
            st.session_state.is_use_tmp_regression_data = False

        # todo: session handling, when exists csv in uploaded_file zone
        with tab2:  # temp data tab
            if st.button('Use Classification Data'):
                st.session_state.is_use_tmp_classification_data = True
                st.session_state.is_use_tmp_regression_data = False
            if st.button('Use Regression Data'):
                st.session_state.is_use_tmp_classification_data = False
                st.session_state.is_use_tmp_regression_data = True
        with tab1:  # uploaded_files of csv tab
            uploaded_files = st.file_uploader("Or Your CSV file", type='csv', accept_multiple_files=False)
            if uploaded_files is not None and (
                    st.session_state.is_use_tmp_classification_data or st.session_state.is_use_tmp_regression_data
            ) and st.button('Use Upload CSV File'):
                st.session_state.is_use_tmp_classification_data = False
                st.session_state.is_use_tmp_regression_data = False

        if st.session_state.is_use_tmp_classification_data and st.session_state.is_use_tmp_regression_data:
            raise ValueError
        elif st.session_state.is_use_tmp_classification_data or st.session_state.is_use_tmp_regression_data:
            st.metric(label='Now Select Data', value='Temp Data')
        elif uploaded_files is not None:
            st.metric(label='Now Select Data', value='Upload CSV File')

        try:
            if st.session_state.is_use_tmp_classification_data or st.session_state.is_use_tmp_regression_data or uploaded_files is not None:
                with st.spinner('Wait for it...'):
                    if st.session_state.is_use_tmp_classification_data:
                        csv_service = CsvService(filepath_or_buffer=get_classification_buffer_data())
                    elif st.session_state.is_use_tmp_regression_data:
                        csv_service = CsvService(filepath_or_buffer=get_regression_buffer_data())
                    elif uploaded_files is not None:
                        csv_service = CsvService(filepath_or_buffer=uploaded_files)
                    else:
                        raise ValueError

                tab1, tab2 = st.tabs(['Data Info', 'sklearn Service'])

                with tab1:  # Check Upload CSV File
                    with st.expander(label='Show Data'):
                        st.table(csv_service.df)

                    with st.expander(label='Show Graph of Data', expanded=True):
                        st.line_chart(csv_service.df)

                    with st.expander(label='Show Diff Column'):
                        st.table(csv_service.calc_diff())

                with tab2:  # sklearn Service
                    predict_type = st.selectbox(
                        label='Predict Type',
                        options=SklearnService.predict_type_list
                    )
                    with st.form(key='sklearn_service_form'):
                        model_name = st.selectbox(
                            label='Model Name',
                            options=SklearnService.model_name_dict[predict_type]
                        )
                        predict_column_name = st.selectbox(
                            label='Predict Column Name',
                            options=csv_service.df.columns
                        )
                        submitted = st.form_submit_button(label='Train And Test')

                        if predict_type is not None and model_name is not None and predict_column_name is not None and submitted:
                            with st.spinner('Wait for it...'):
                                sklearn_service = SklearnService(predict_type=predict_type, model_name=model_name)
                                sklearn_service.main(df=csv_service.df, predict_column_name=predict_column_name)

                            # tabs after fit model
                            result_tab, feature_importance_tab = st.tabs(['Result', 'Feature Importance'])
                            with result_tab:
                                st.metric(label='Result Score',
                                          value=sklearn_service.result_score)
                                st.markdown('Test Data And Predict Value')
                                st.table(data=sklearn_service.submission_df)
                            with feature_importance_tab:
                                st.table(data=sklearn_service.model.feature_importances_)

        except Exception as e:
            logger.error(f'ERROR: {uploaded_files=}')

    def stock_service(self):
        st.title('Stock service')

        stock_service = StockService()

        tab1, tab2, tab3 = st.tabs(['Check Signal', 'Check Per Ticker', 'Check Stock Value'])

        with tab1:  # check_signal
            st.info(f'Ticker: {", ".join(stock_service.ticker_list)}')
            with st.form('check_signal_form'):

                c1, c2, c3, c4 = st.columns([0.25, 0.25, 0.25, 0.25])

                with c1:
                    start_date = st.date_input('Start date', datetime.date(2020, 1, 1))
                with c2:
                    end_date = st.date_input('End date')
                with c3:
                    how_date = st.date_input('How long in the past do you check signals?',
                                             datetime.date.today() - timedelta(7))

                if start_date > end_date:
                    st.error('Please start_date before end_date.')
                    is_check_signal_start_disabled: bool = True
                elif not (start_date < how_date < end_date):
                    st.error('Please start_date before how_date. '
                             'And end_date after how_date.')
                    is_check_signal_start_disabled: bool = True
                else:
                    is_check_signal_start_disabled: bool = False

                with c4:
                    st.markdown('Get Signal Result')
                    submitted = st.form_submit_button(label='GET')

                if not is_check_signal_start_disabled and submitted:
                    with st.spinner('Wait for it...'):
                        stock_service.signal_check_main(start_date=start_date, end_date=end_date, how_date=how_date)
                    st.success('Success')
                    st.info(f'{start_date=}, {end_date=}, {how_date=}')
                    st.table(stock_service.get_result_signal_df())

        with tab2:  # check_per_ticker
            with st.form(key='check_per_ticker_form'):
                option = st.selectbox(
                    label='Ticker',
                    options=stock_service.ticker_list
                )
                submitted = st.form_submit_button(label='GET')

            if option is not None and submitted:
                with st.spinner('Wait for it...'):
                    stock_service.main(ticker=option)
                st.success('Success')
                st.info(f'ticker: {option}')
                with st.container():
                    st.markdown('## Close Value')
                    st.pyplot(stock_service.fig)
                with st.container():
                    st.markdown('## RCI')
                    st.pyplot(stock_service.rci_fig)
                with st.container():
                    st.markdown('## RSI')
                    st.pyplot(stock_service.rsi_fig)
                with st.container():
                    st.markdown('## MACD')
                    st.pyplot(stock_service.macd_fig)

        with tab3:  # check_stock_value
            is_view_df: bool = False

            with st.form(key='check_stock_value_form'):

                c1, c2, c3, c4 = st.columns([0.25, 0.25, 0.25, 0.25])

                with c1:
                    option = st.selectbox(
                        label='Ticker',
                        options=stock_service.ticker_list
                    )
                with c2:
                    start_date = st.date_input('Start date', datetime.date(2020, 1, 1))
                with c3:
                    end_date = st.date_input('End date')
                    if start_date > end_date:
                        st.error('Please start_date before end_date.')
                        is_check_stock_value_start_disabled: bool = True
                    else:
                        is_check_stock_value_start_disabled: bool = False
                with c4:
                    st.markdown('Get Stock Value')
                    submitted = st.form_submit_button(label='GET')

                if option is not None and not is_check_stock_value_start_disabled and submitted:
                    with st.spinner('Wait for it...'):
                        df = stock_service.get_stock(code=option, start_date=start_date, end_date=end_date)
                    st.info(f'Ticker: {option}')
                    st.line_chart(df.drop(columns=['Volume']))
                    st.bar_chart(df['Volume'])
                    is_view_df = True

            with st.expander(label='Stock Value Table', expanded=True):
                if is_view_df:
                    st.table(df)

    def calc_service(self):
        st.title('Calc Service')
        calc_service = CalcService()

        variables = st.text_input('variables')
        input = st.text_input('eval')
        try:
            result = None
            if not isinstance(input, str):
                input = str(input)
            if input is not None and variables is not None:
                result = calc_service.get_eval(input, variables)
            elif input is not None:
                result = calc_service.get_eval(input)
            if result is None:
                raise ValueError
        except Exception as e:
            st.metric(label='Variables', value=variables)
            st.metric(label='Text Input', value=input)
        else:
            st.metric(label='Variables', value=variables)
            st.metric(label='Result', value=result)

    def markdown_service(self):
        st.title('Markdown Service')
        input = st.text_area('markdown text: ')
        if input is not None:
            st.markdown(input)

    def notion_service(self):
        st.title('Notion Service')

        try:
            notion_service = NotionService(access_token=st.secrets['notion_access_token'])
            if st.button('GET'):
                with st.spinner('Wait for it...'):
                    res = notion_service.show_database(database_id=st.secrets['notion_database_id'])
                st.table(res)
                st.json(notion_service.result_dict)
        except Exception as e:
            logger.error(e)
            st.error('access_token error')

    def etc_service(self):
        st.markdown("""
        |Sub Page          |Functions           |
        |------------------|--------------------|
        |image_service     |upload              |
        |                  |view                |
        |csv_service       |upload csv file     |
        |                  |use temp data       |
        |                  |Data Info           |
        |                  |sklearn service     |
        |stock_service     |Check Signal        |
        |                  |CHeck Per Ticker    |
        |                  |CHeck Stock Value   |
        |etc               |*this page*         |
        """)
