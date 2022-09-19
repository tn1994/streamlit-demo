import logging

import streamlit as st

from views.sidebar import Sidebar
from services.hash_service import check_hashes

logger = logging.getLogger(__name__)


class APP:
    st: st = st
    env: str = st.secrets["env"]
    hashed_text: str = st.secrets['hashed_text']

    if 'is_authorization' not in st.session_state:
        st.session_state.is_authorization = False

    def __init__(self):
        if self.env not in ['prod', 'develop']:
            raise ValueError

    def main(self):
        option = self.st.sidebar.selectbox(
            label='main page:',
            options=['login', 'main'],
            index=1 if self.st.session_state.is_authorization else 0
        )
        if 'login' == option and not self.st.session_state.is_authorization:
            self._top_page()
            self.st.session_state.is_authorization = self._password()
        if 'login' == option and self.st.session_state.is_authorization:
            self.st.sidebar.info('Switch to "Main" from the select box in the sidebar.')
        if 'main' == option and self.st.session_state.is_authorization:
            self._sidebar()

        # view Authorization in sidebar
        self.st.sidebar.markdown('### Now Your Authorization:')
        match self.st.session_state.is_authorization:
            case True:
                self.st.sidebar.success('Authorization')
            case False:
                self.st.sidebar.warning('Unauthenticated')
            case _:
                raise ValueError(f'{self.st.session_state.is_authorization=}')

    def _password(self) -> bool:
        with self.st.form(key='password'):
            password = self.st.text_input(label='Please enter your password.', type='password', value='0')
            submit_button = self.st.form_submit_button(label='Submit')
        if submit_button:
            is_authorization = check_hashes(password=password, hashed_text=self.hashed_text)
            if is_authorization:
                return True
            else:
                self.st.write('Please Re-Input and Re-Submit.')
        return False

    def _top_page(self):
        self._title()
        self._write()

    def _title(self):
        self.st.title('streamlit-demo')

    def _write(self):
        self.st.write('check write')

    def _sidebar(self):
        sidebar = Sidebar(st=self.st)
        pass


def main():
    app = APP()
    app.main()


if __name__ == '__main__':
    main()
