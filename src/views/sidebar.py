import io
import uuid
import logging
import datetime
import requests
import traceback
from datetime import timedelta

from PIL import Image
import streamlit as st

logger = logging.getLogger(__name__)

try:
    from ..services.image_service import ImageService
    from ..services.image_service import SearchImageService
    from ..services.image_service import DownloadImageService
    from ..services.image_services.pinterest_service import PinterestService
    from ..services.nlp_services.gensim_service import GensimService
    from ..services.csv_service import CsvService
    from ..services.csv_service import get_classification_buffer_data
    from ..services.csv_service import get_regression_buffer_data
    from ..services.aws_service import AWSService
    from ..services.hugging_face_service import HuggingFaceService
    from ..services.hugging_face_service import HuggingFaceBuiltInService
    from ..services.keybert_service import KeyBERTService
    from ..services.stock_service import StockService
    from ..services.stock_service import color_survived
    from ..services.calc_service import CalcService
    from ..services.notion_service import NotionService
    from .notion_pinterest_view import NotionPinterestView
    from ..services.version_service import VersionService
    from ..services.sklearn_service import SklearnService
except ImportError:
    logger.info('check: ImportError')  # todo: fix import error
    from services.image_service import ImageService
    from services.image_service import SearchImageService
    from services.image_service import DownloadImageService
    from services.image_services.pinterest_service import PinterestService
    from services.nlp_services.gensim_service import GensimService
    from services.csv_service import CsvService
    from services.csv_service import get_classification_buffer_data
    from services.csv_service import get_regression_buffer_data
    from services.aws_service import AWSService
    from services.hugging_face_service import HuggingFaceService
    from services.hugging_face_service import HuggingFaceBuiltInService
    from services.keybert_service import KeyBERTService
    from services.stock_service import StockService
    from services.stock_service import color_survived
    from services.calc_service import CalcService
    from services.notion_service import NotionService
    from views.notion_pinterest_view import NotionPinterestView
    from services.version_service import VersionService
    from services.sklearn_service import SklearnService


class Sidebar:  # todo: refactor

    def __init__(self):
        self.service_dict = {
            'image_service': self.image_service,
            'pinterest_service': self.pinterest_service,
            'gensim_service': self.gensim_service,
            'csv_service': self.csv_service,
            'aws_service': self.aws_service,
            'hugging_face_service': self.hugging_face_service,
            'hugging_face_demo_service': self.hugging_face_demo_service,
            'keybert_service': self.keybert_service,
            'stock_service': self.stock_service,
            'calc_service': self.calc_service,
            'markdown_service': self.markdown_service,
            'notion_service': self.notion_service,
            'notion_pinterest_service': self.notion_pinterest_service,
            'version_service': self.version_service,
            'etc': self.etc_service
        }

    def main(self):
        radio_value = st.sidebar.radio('Sub Page', self.service_dict.keys())
        if radio_value:
            select_service = self.service_dict[radio_value]
            select_service()

    def image_service(self):
        st.title('Image service')

        tab1, tab2, tab3 = st.tabs(['Upload Image Service', 'Search Image Service', 'Download as Zip'])

        with tab1:
            uploaded_file = st.file_uploader('Choose a image file.', type=['jpeg', 'png'])
            try:
                if uploaded_file is not None:
                    image_service = ImageService()
                    image_service.set_image(fp=uploaded_file)
                    st.image(image_service.image, caption='upload image', use_column_width=True)
            except Exception as e:
                logger.error(f'ERROR: {uploaded_file=}')

        with tab2:
            try:
                url: str = st.text_input(label='search url')
                if st.button('Show Images') and 0 != len(url):
                    # ref:https://cafe-mickey.com/python/streamlit-5/
                    search_image = SearchImageService()
                    img_list: list = search_image.get_img_link(url=url)

                    num = 1
                    col = st.columns(num)
                    if 0 != len(img_list):
                        for idx, img_link in enumerate(img_list):
                            with col[idx % num]:
                                st.image(img_list[idx], use_column_width=True)

            except Exception as e:
                logger.error(e)
                logger.error(f'ERROR: Search Image Service')

        with tab3:
            try:
                download_image_service = DownloadImageService(cx=st.secrets['google_custom_search_api']['cx'],
                                                              key=st.secrets['google_custom_search_api']['key'])

                with st.form(key='download_image_service_form'):

                    select_query: str = st.selectbox(label='Select Query', options=download_image_service.query_list)
                    query: str = st.text_input(label='Other Query')
                    num_images: int = st.slider('Num of Images', 0, 100, 25)
                    submitted = st.form_submit_button(label='Setup Download')

                if select_query is not None and num_images is not None and submitted:
                    with st.spinner('Wait for it...'):
                        with io.BytesIO() as buffer:  # ref: https://discuss.streamlit.io/t/download-zipped-json-file/22512/5
                            _query: str = query if 0 != len(query) else select_query
                            zipfile = download_image_service.download_images_as_zipfile(buffer=buffer, query=_query,
                                                                                        num=num_images)
                            buffer.seek(0)

                            if zipfile is not None:
                                st.download_button(label='Download Images as ZipFile',
                                                   data=zipfile,
                                                   file_name='images.zip',
                                                   mime='application/zip')
            except Exception as e:
                logger.error(e)
                traceback.print_exc()

    def pinterest_service(self):
        pinterest_view = PinterestView()
        pinterest_view.main()

    def gensim_service(self):
        gensim_view = GensimView()
        gensim_view.main()

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

    def aws_service(self):
        st.title('AWS Service')

        try:
            aws_service = AWSService(
                aws_access_key_id=st.secrets['aws_service']['access_key_id'],
                aws_secret_access_key=st.secrets['aws_service']['secret_access_key'],
                region_name=st.secrets['aws_service']['region_name']
            )

            tab1, tab2 = st.tabs(['Billing', 'S3'])

            with tab1:
                with st.expander('All Billing', expanded=True):
                    if st.button('Show Billing'):
                        with st.spinner('Wait for it...'):
                            billing = aws_service.get_total_billing()
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            from decimal import Decimal, ROUND_HALF_UP
                            _billing = Decimal(billing["billing"]).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                            st.metric(label=f'Billing', value=f'${_billing}-')
                        with c2:
                            st.metric(label=f'From', value=f'{billing["start"]}')
                        with c3:
                            st.metric(label=f'To', value=f'{billing["end"]}')

                with st.expander('Billing Per Service', expanded=True):
                    if st.button('Show Per Service'):
                        import pandas as pd
                        with st.spinner('Wait for it...'):
                            billing = aws_service.get_service_billings()
                        _billing = pd.DataFrame(billing).astype({'service_name': str, 'billing': float})
                        st.bar_chart(_billing, x='service_name', y='billing')
                        st.table(_billing)

            with tab2:
                if st.button('Show List Buckets'):
                    with st.spinner('Wait for it...'):
                        list_buckets = aws_service.get_list_buckets()
                    st.table(list_buckets['Buckets'])
        except Exception as e:
            logger.error(e)
            st.error('aws access error')

    def hugging_face_service(self):
        hugging_face_view = HuggingFaceView()
        hugging_face_view.main()

    def hugging_face_demo_service(self):
        hugging_face_demo_view = HuggingFaceDemoView()
        hugging_face_demo_view.main()

    def keybert_service(self):
        keybert_view = KeyBERTView()
        keybert_view.main()

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
        """
        ref: https://zenn.dev/megane_otoko/articles/086_streamlit_session_state
        :return:
        """
        st.title('Calc Service')
        calc_service = CalcService()

        if 'calc_service_variable_unique_id' not in st.session_state:
            st.session_state.calc_service_variable_unique_id = [uuid.uuid1()]

        c1, c2, _ = st.columns([0.2, 0.2, 0.6])
        with c1:
            if st.button('Add Variable'):
                st.session_state.calc_service_variable_unique_id.append(uuid.uuid1())
        with c2:
            if st.button("Delete Variable") and 1 < len(st.session_state.calc_service_variable_unique_id):
                st.session_state.calc_service_variable_unique_id.pop(-1)

        variables: str = ''
        for unique_id in st.session_state.calc_service_variable_unique_id:
            variable = st.text_input(label='variables', key=unique_id, value='a = 1_000_000')
            if variable is not None:
                match len(variables):
                    case 0:
                        variables = variable
                    case _:
                        variables = f'{variables}; {variable}'

        input = st.text_input(label='eval', value='a / 3_000_000')

        try:
            result = None
            if input is not None and 0 != len(input) and variables is not None:
                logger.info(f'{variables=}')
                result = calc_service.get_eval(input, variables)
            elif input is not None and 0 != len(input):
                result = calc_service.get_eval(input)
        except Exception as e:
            st.metric(label='Variables', value=variables)
            st.metric(label='Text Input', value=input)
            st.error(e)
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
            notion_service = NotionService(access_token=st.secrets['notion_service']['access_token'])
            if st.button('GET'):
                with st.spinner('Wait for it...'):
                    res = notion_service.show_database(database_id=st.secrets['notion_service']['database_id'])
                st.table(res)
                st.json(notion_service.result_dict)
        except Exception as e:
            logger.error(e)
            st.error('access_token error')

    def notion_pinterest_service(self):
        notion_pinterest_view = NotionPinterestView()
        notion_pinterest_view.main()

    def version_service(self):
        st.title('Version Service')
        version_service = VersionService()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label='Python Version', value=version_service.get_python_version())
        with c2:
            st.metric(label='Pip Version', value=version_service.get_pip_version())
        with c3:
            st.metric(label='Streamlit Version',
                      value=version_service.get_library_version(library_name='streamlit'))
        st.download_button(label='Download requirements.txt',
                           data=version_service.get_pip_list(format='freeze'),
                           file_name='requirements.txt',
                           mime='text/txt')
        with st.spinner('Wait for it...'):
            pip_list = version_service.get_pip_list(format='json')
        with st.expander('Pip List', expanded=True):
            st.table(pip_list)

    def etc_service(self):
        st.title('Sub Page Info')
        st.markdown("""
        |Sub Page                 |Functions                                            |Remarks             |
        |-------------------------|-----------------------------------------------------|--------------------|
        |image_service            |upload                                               |                    |
        |                         |view                                                 |                    |
        |csv_service              |upload csv file                                      |                    |
        |                         |use temp data                                        |                    |
        |                         |Data Info                                            |                    |
        |                         |sklearn service                                      |                    |
        |aws_service              |Billing - All Billing                                |must set st.secrets |
        |                         |Billing - Billing Per Service                        |must set st.secrets |
        |                         |S3 - Show List Buckets                               |must set st.secrets |
        |hugging_face_service     |Hugging Face Inference API - Text Generation         |must set st.secrets |
        |                         |Hugging Face Inference API - Fill Mask               |must set st.secrets |
        |                         |Built in - Text Generation                           |                    |
        |                         |Built in - Fill Mask                                 |                    |
        |stock_service            |Check Signal                                         |                    |
        |                         |CHeck Per Ticker                                     |                    |
        |                         |CHeck Stock Value                                    |                    |
        |calc_service             |Calc Eval                                            |                    |
        |markdown_service         |Markdown text input                                  |                    |
        |                         |Markdown text view                                   |                    |
        |                         |CHeck Stock Value                                    |                    |
        |notion_service           |Get Notion Database                                  |must set st.secrets |
        |version_service          |Get requirements.txt                                 |                    |
        |                         |View Python Version                                  |                    |
        |                         |View Pip Version                                     |                    |
        |                         |View Streamlit Version                               |                    |
        |etc                      |*this page*                                          |                    |
        """)


class PinterestView:
    title: str = 'Pinterest Service'

    def main(self):
        st.title(self.title)

        pinterest_service = PinterestService()

        with st.form(key='pinterest_service_form'):
            select_query: str = st.selectbox(label='Select Query', options=pinterest_service.query_list)
            query: str = st.text_input(label='Other Query')
            num_pins: int = st.slider('Num of Images', 0, 100, 25)
            submitted = st.form_submit_button(label='Search')

        if 0 != len(select_query) and num_pins is not None and submitted:
            with st.spinner('Wait for it...'):
                _query: str = query if 0 != len(query) else select_query
                pinterest_service.search(query=_query, num_pins=num_pins)

            with st.expander(label='Show Pins', expanded=True):
                num = 3
                col = st.columns(num)
                if 0 != len(pinterest_service.image_info_list):
                    for idx, img_link in enumerate(pinterest_service.image_info_list):
                        with col[idx % num]:
                            st.image(pinterest_service.image_info_list[idx], use_column_width=True)


class GensimView:
    title: str = 'Gensim Service'

    def __init__(self):
        self.gensim_service = GensimService()
        self.gensim_service.main()

    def main(self):
        st.title(self.title)

        st.table(self.gensim_service.most_similar_list)
        st.write(f'Positive: {self.gensim_service.random_positive_list[self.gensim_service.random_positive_idx]}')
        st.write(f'Negative: {self.gensim_service.random_negative_list[self.gensim_service.random_negative_idx]}')

        with st.form(key='pinterest_service_form'):
            # select_query: str = st.selectbox(label='Select Query', options=pinterest_service.query_list)
            query: str = st.text_input(label='Negative Word')
            # num_pins: int = st.slider('Num of Images', 0, 100, 25)
            submitted = st.form_submit_button(label='Answer')

        if 0 != len(query) and submitted:
            with st.spinner('Wait for it...'):
                is_correct_answer: bool = self.gensim_service.is_correct_answer(word=query)

                if is_correct_answer:
                    st.success('Correct!')
                elif not is_correct_answer:
                    st.warning('Not Correct...')


class HuggingFaceView:
    title: str = 'Hugging Face Service'
    main_tab_list: list = ['Hugging Face Inference API', 'Built in']

    def main(self):
        st.title(self.title)
        tab_api, tab_built_in = st.tabs(self.main_tab_list)
        try:
            with tab_api:  # API tab
                hugging_face_service = HuggingFaceService(
                    access_token=st.secrets['hugging_face_service']['access_token'])

                tab1, tab2 = st.tabs(hugging_face_service.task_list)
                with tab1:
                    select_model_name = st.selectbox(label='Select Model',
                                                     options=hugging_face_service.text_generation_model_id_list)
                    with st.form(key='task_generation_form'):
                        payload = st.text_input(label='Query',
                                                value=hugging_face_service.get_example_value(
                                                    model_name=select_model_name))
                        submitted = st.form_submit_button(label='Inference')

                        if select_model_name is not None and 0 != len(payload) and submitted:
                            with st.spinner(text='Wait for it...'):
                                result = hugging_face_service.query(payload=payload, model_name=select_model_name)
                            st.info(f'Input Text: \n\n {payload}')
                            st.info(f'Result: \n\n {result[0]["generated_text"]}')
                with tab2:
                    select_model_name = st.selectbox(label='Select Model',
                                                     options=hugging_face_service.fill_mask_model_id_list)
                    with st.form(key='fill_mask_form'):
                        payload = st.text_input(label='Query',
                                                value=hugging_face_service.get_example_value(
                                                    model_name=select_model_name))
                        submitted = st.form_submit_button(label='Inference')

                        if select_model_name is not None and 0 != len(payload) and submitted:
                            with st.spinner(text='Wait for it...'):
                                result = hugging_face_service.query(payload=payload, model_name=select_model_name)
                            st.info(f'Input Text: \n\n {payload}')
                            st.markdown('### Result')
                            st.table(result)
        except Exception as e:
            logger.error(e)
            st.error('huggingface_access_token error')

        try:
            with tab_built_in:  # Built in tab
                hugging_face_built_in_service = HuggingFaceBuiltInService()

                tab1, tab2 = st.tabs(hugging_face_built_in_service.task_list)

                with tab1:
                    select_model_name = st.selectbox(label='Select Model',
                                                     options=hugging_face_built_in_service.text_generation_model_id_list)
                    with st.form(key='task_generation_built_in_form'):
                        payload = st.text_input(label='Query',
                                                value=hugging_face_built_in_service.get_example_value(
                                                    model_name=select_model_name))
                        submitted = st.form_submit_button(label='Inference')

                        if select_model_name is not None and 0 != len(payload) and submitted:
                            with st.spinner('Wait for it...'):
                                result = hugging_face_built_in_service.inference(payload=payload,
                                                                                 model_name=select_model_name)
                            st.info(f'Input Text: \n\n {payload}')
                            for sentence in result:
                                st.info(f'Result: \n\n {sentence}')

                with tab2:
                    select_model_name = st.selectbox(label='Select Model',
                                                     options=hugging_face_built_in_service.fill_mask_model_id_list)
                    with st.form(key='fill_mask_built_in_form'):
                        payload = st.text_input(label='Query',
                                                value=hugging_face_built_in_service.get_example_value(
                                                    model_name=select_model_name))
                        submitted = st.form_submit_button(label='Inference')

                        if select_model_name is not None and 0 != len(payload) and submitted:
                            with st.spinner('Wait for it...'):
                                result = hugging_face_built_in_service.inference(payload=payload,
                                                                                 model_name=select_model_name)
                            st.info(f'Input Text: \n\n {payload}')
                            st.markdown('### Result')
                            st.table(result)

        except Exception as e:
            logger.error(e)
            st.error('Built in error')


class HuggingFaceDemoView:
    title: str = 'Hugging Face Demo Service'

    def main(self):
        st.title(self.title)

        try:
            self._view_select_keybert_service()
        except Exception as e:
            logger.error(e)
            st.error('Select KeyBERT error')

    def _view_select_keybert_service(self):
        try:
            hugging_face_built_in_service = HuggingFaceBuiltInService()

            select_model_name = st.selectbox(label='Select Model',
                                             options=hugging_face_built_in_service.text_generation_model_id_list)
            with st.form(key='task_generation_built_in_form'):
                payload = st.text_input(label='Query',
                                        value=hugging_face_built_in_service.get_example_value(
                                            model_name=select_model_name))
                submitted = st.form_submit_button(label='Inference')

                if select_model_name is not None and 0 != len(payload) and submitted:
                    with st.spinner('Wait for it...'):
                        result = hugging_face_built_in_service.inference(payload=payload,
                                                                         model_name=select_model_name)
                    st.info(f'Input Text: \n\n {payload}')
                    for sentence in result:
                        st.info(f'Result: \n\n {sentence}')
        except Exception as e:
            logger.error(e)
            raise e


class KeyBERTView:
    title: str = 'KeyBERT Service'
    main_tab_list: list = ['Select Model KeyBERT Service', 'Base KeyBERT Service']

    def main(self):
        st.title(self.title)
        tab_select_model_keybert, tab_base_keybert = st.tabs(self.main_tab_list)

        try:
            with tab_select_model_keybert:
                self._view_select_keybert_service()
        except Exception as e:
            logger.error(e)
            st.error('Select KeyBERT error')

        try:
            with tab_base_keybert:
                self._view_base_keybert_service()
        except Exception as e:
            logger.error(e)
            st.error('Base KeyBERT error')

    def _view_select_keybert_service(self):
        try:
            keybert_service = KeyBERTService()

            select_model_name = st.selectbox(label='Select Model', options=keybert_service.word_embedding_model_list)
            with st.form(key='select_keybert_form'):
                payload = st.text_area(label='Doc',
                                       value=keybert_service.get_example_value(model_name=select_model_name))
                add_tokens = st.text_area(label='Add Token',
                                          value=keybert_service.get_example_add_tokens(model_name=select_model_name))
                submitted = st.form_submit_button(label='Inference')

                if select_model_name is not None and 0 != len(payload) and submitted:
                    with st.spinner('Wait for it...'):
                        if add_tokens is not None:
                            result = keybert_service.main(model_name=select_model_name, payload=payload,
                                                          add_tokens=add_tokens)
                        else:
                            result = keybert_service.main(model_name=select_model_name, payload=payload)
                    st.markdown('### Result')
                    st.table(result)
        except Exception as e:
            logger.error(e)
            raise e

    def _view_base_keybert_service(self):
        try:
            keybert_service = KeyBERTService()

            with st.form(key='base_keybert_form'):
                payload = st.text_area(label='Doc')
                submitted = st.form_submit_button(label='Inference')

                if 0 != len(payload) and submitted:
                    with st.spinner('Wait for it...'):
                        result = keybert_service.base_extract_keywords(payload=payload)
                    st.markdown('### Result')
                    st.table(result)
        except Exception as e:
            logger.error(e)
            raise e
