from dataclasses import dataclass


@dataclass
class NotionBaseModel:
    column_name_list: list = None
    tag_list: list = None

    def __init__(self):
        if self.column_name_list is None:
            raise AttributeError
        if self.tag_list is None:
            raise AttributeError
