import json
import logging
import requests

from notion_client import Client

try:
    from ...models.notion_pinterest_model import PinterestModel
except ImportError as e:
    from models.notion_pinterest_model import PinterestModel

logger = logging.getLogger(__name__)


class NotionBaseService:
    def __init__(self, access_token: str):
        """
        ref:
        https://developers.notion.com/reference/intro
        https://blog.rmc-8.com/2021/06/using-notion-api-with-python.html
        https://zenn.dev/team_zenn/articles/117424abb5605b
        :param access_token:
        """
        self.notion = Client(auth=access_token)

        if not isinstance(access_token, str):
            raise TypeError

        self.headers = {
            "Accept": "application/json",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + access_token
        }


class NotionPinterestService(NotionBaseService):
    result_dict = {}

    column_name_list = PinterestModel().column_name_list
    tag_list = PinterestModel().tag_list

    def __init__(self, access_token: str):
        super(NotionPinterestService, self).__init__(access_token=access_token)

    def query(self, query: str, database_id: str):
        try:
            my_page = self.notion.databases.query(
                **{
                    "database_id": database_id,
                    "filter": {
                        "property": "TAG",
                        "multi_select": {
                            "contains": query,
                        },
                    },
                }
            )
            result_list: list = self.convert_data(result=my_page['results'])
        except Exception as e:
            raise e
        else:
            return result_list

    def convert_data(self, result: list):
        result_list: list = []
        for item in result:
            tmp_dict = {}
            for key in self.column_name_list:  # todo: change to item['properties'].keys() ?
                attribute = item['properties'][key]
                match attribute['type']:
                    case 'number':
                        value = attribute['number']
                    case 'title':
                        if len(attribute['title']):
                            value = attribute['title'][0]['text']['content']
                        else:
                            value = None
                    case 'rich_text':
                        value = attribute['rich_text'][0]['text']['content']
                    case 'multi_select':
                        value = attribute['multi_select'][0]['name']
                    case _:
                        value = None
                        logger.info(f'{attribute.type=}')
                tmp_dict[key] = value
            result_list.append(tmp_dict)
        return result_list

    def show_database(self, database_id: str):
        try:
            url = f'https://api.notion.com/v1/databases/{database_id}/query'
            response = requests.post(url, headers=self.headers)
            if 200 != response.status_code:
                raise
            if 'json' in response.headers.get('content-type'):
                self.result_dict = response.json()
                result = self.result_dict["results"]
                result_list: list = self.convert_data(result=result)
            else:
                raise
        except Exception as e:
            logger.error(e)
        else:
            return result_list

    def insert_item(self, database_id: str, link_id: str, tag: str, hash: str = None):
        try:
            url = f'https://api.notion.com/v1/pages'
            data = {
                'parent': {'database_id': database_id},
                'properties': {
                    'LINK_ID': {
                        'title': [{'text': {'content': link_id}}]
                    },
                    'TAG': {
                        'multi_select': [{'name': tag}]
                    },
                    'HASH': {
                        'rich_text': [{'text': {'content': hash}}]
                    },
                },
            }
            data = json.dumps(data)
            response = requests.post(url, headers=self.headers, data=data)
            if 200 != response.status_code:
                logger.error(f'{response.content=}')
                raise
            if 'json' in response.headers.get('content-type'):
                self.result_dict = response.json()
        except Exception as e:
            logger.error(e)
            raise e
        else:
            return self.result_dict
