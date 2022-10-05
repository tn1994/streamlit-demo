import logging

import streamlit as st

logger = logging.getLogger(__name__)

try:
    from ..services.hash_service import make_hashes
    from ..services.notion_services.pinterest_service import NotionPinterestService
except ImportError:
    from services.hash_service import make_hashes
    from services.notion_services.pinterest_service import NotionPinterestService


class NotionPinterestView:
    title: str = 'Notion Pinterest Service'
    database_id: str = None

    def main(self):
        st.title(self.title)

        try:
            self.database_id: str = st.secrets['notion_pinterest_service']['database_id']

            tab1, tab2, tab3 = st.tabs(['GET', 'POST', 'READ'])

            notion_pinterest_service = NotionPinterestService(
                access_token=st.secrets['notion_pinterest_service']['access_token'])

            with tab1:
                if st.button('Select All'):
                    with st.spinner('Wait for it...'):
                        res = notion_pinterest_service.show_database(database_id=self.database_id)
                    st.table(res)
                    st.json(notion_pinterest_service.result_dict)

            with tab2:
                with st.form(key='notion_pinterest_insert_service_form'):
                    link_id: str = st.text_input(label='Link ID')
                    tag: str = st.selectbox(label='Select Tag', options=notion_pinterest_service.tag_list)
                    hash: str = make_hashes(password=link_id)
                    submitted = st.form_submit_button(label='CREATE')

                if 0 != len(link_id) and 0 != len(tag) and 0 != len(hash) and submitted:
                    with st.spinner('Wait for it...'):
                        notion_pinterest_service.insert_item(database_id=self.database_id, link_id=link_id, tag=tag,
                                                             hash=hash)
                    st.json(notion_pinterest_service.result_dict)

            with tab3:
                with st.form(key='notion_pinterest_search_service_form'):
                    tag: str = st.selectbox(label='Search Tag', options=notion_pinterest_service.tag_list)
                    submitted = st.form_submit_button(label='SEARCH')

                if 0 != len(tag) and submitted:
                    with st.spinner('Wait for it...'):
                        res = notion_pinterest_service.query(query=tag, database_id=self.database_id)
                    st.table(res)
                    st.json(res)

        except Exception as e:
            logger.error(e)
            st.error('access_token error')
