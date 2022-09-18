import os
import streamlit as st

from views.sidebar import Sidebar

from services.hash_service import check_hashes


class APP:
    st: st = st
    env: str = os.environ['env']
    hashed_text: str = os.environ['hashed_text']

    def __init__(self):
        pass

    def main(self):
        match self.env:
            case 'develop':
                is_success: bool = True
            case _:
                is_success: bool = self._password()

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
        self.st.title('check title')

    def _write(self):
        self.st.write('check write')

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
                pass
            case _:
                pass


def main():
    app = APP()
    app.main()


if __name__ == '__main__':
    main()
