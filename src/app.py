import streamlit as st

from views.sidebar import Sidebar


class APP:
    st: st = st

    def __init__(self):
        pass

    def main(self):
        self._title()
        self._write()
        self._sidebar()

    def _title(self):
        self.st.title('check title')

    def _write(self):
        self.st.write('check write')

    def _sidebar(self):
        sidebar = Sidebar(st=self.st)


def main():
    app = APP()
    app.main()


if __name__ == '__main__':
    main()
