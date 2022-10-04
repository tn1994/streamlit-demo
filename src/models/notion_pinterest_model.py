from .notion_base_model import NotionBaseModel


class PinterestModel(NotionBaseModel):
    column_name_list: list = ['LINK_ID', 'TAG', 'HASH']
    tag_list: list = ['a', 'b', 'c']
