class Sidebar:
    def __init__(self, st):
        self.st = st
        self.main()

    def main(self):
        self.st.sidebar.markdown('check sidebar')
        self._check_radio()

    def _check_radio(self):
        radio_value_dict = {
            'a': self.st.write,
            'b': self.st.write,
            'c': self.st.write
        }
        radio_value = self.st.sidebar.radio('check_radio', radio_value_dict.keys())

        if radio_value:
            radio_value_dict[radio_value](f'checkd: {radio_value=}')
