import logging

import streamlit as st

from views.sidebar import Sidebar
from views.about_me import AboutMe
from services.hash_service import check_hashes
from services.utils.analysis_memory import get_memory_state_percent

logger = logging.getLogger(__name__)


class APP:
    env: str = st.secrets["env"]
    hashed_text: str = st.secrets['hashed_text']

    if 'is_authorization' not in st.session_state:
        st.session_state.is_authorization = False

    def __init__(self):
        if self.env not in ['prod', 'develop']:
            raise ValueError
        st.set_page_config(
            page_title='tn1994/streamlit-demo',
            layout='wide'
        )

    def main(self):
        option = st.sidebar.selectbox(
            label='Main Page',
            options=['login', 'main'],
            index=1 if st.session_state.is_authorization else 0
        )

        if 'login' == option:
            self._top_page()

            if 90 > get_memory_state_percent():
                if not st.session_state.is_authorization:
                    st.session_state.is_authorization = self._password()
                if st.session_state.is_authorization:
                    st.info('Switch to "Main" from the select box in the sidebar.')

                # view Authorization
                st.markdown('### Now Your Authorization:')
                match st.session_state.is_authorization:
                    case True:
                        st.success('Authorization')
                    case False:
                        st.warning('Unauthenticated')
                    case _:
                        raise ValueError(f'{st.session_state.is_authorization=}')
            else:
                st.error('Memory Over Error')

        if 'main' == option and st.session_state.is_authorization:
            self._sidebar()

    def _password(self) -> bool:
        with st.form(key='password'):
            password = st.text_input(label='Please enter your password.', type='password', value='0')
            submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            is_authorization = check_hashes(password=password, hashed_text=self.hashed_text)
            if is_authorization:
                return True
            else:
                st.write('Please Re-Input and Re-Submit.')
        return False

    def _top_page(self):
        self._title()
        AboutMe.main()

    def _title(self):
        st.title('streamlit-demo')

    def _write(self):
        st.write('check write')

    def _sidebar(self):
        sidebar = Sidebar()
        sidebar.main()


def main():
    app = APP()
    app.main()


if __name__ == '__main__':
    main()
