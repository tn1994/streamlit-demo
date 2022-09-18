import os
import logging

import streamlit as st

from views.sidebar import Sidebar
from services.hash_service import check_hashes

logger = logging.getLogger(__name__)


class APP:
    st: st = st
    env: str = os.environ['env']
    hashed_text: str = os.environ['hashed_text']

    def __init__(self):
        if self.env not in ['prod', 'develop']:
            raise ValueError

    def main(self):
        match self.env:
            case 'develop':
                is_success: bool = True
            case _:
                if self._password() is not False:
                    is_success: bool = True
                else:
                    is_success: bool = False

        if is_success:
            self._title()
            self._write()
            self._sidebar()
        else:
            self.st.write('No Access')

    def _password(self):
        password = st.sidebar.text_input("Please enter your password.", type='password')
        return check_hashes(password=password, hashed_text=self.hashed_text)

    def _title(self):
        self.st.title('streamlit-demo')

    def _write(self):
        # self.st.write('check write')
        pass

    def _sidebar(self):
        sidebar = Sidebar(st=self.st)

        match sidebar.radio_value:
            case 'image_service':
                self.st.title('image service')
                sidebar.image_service()
            case 'csv_service':
                self.st.title('csv service')
                sidebar.csv_service()
            case 'etc':
                sidebar.etc_service()
            case _:
                pass


def main():
    app = APP()
    app.main()


if __name__ == '__main__':
    main()
