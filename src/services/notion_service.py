import logging
import requests

logger = logging.getLogger(__name__)


class NotionModel:
    column_name_list: list = ['Name', 'Number', 'Tag']


class NotionService:
    result_dict = {}

    def __init__(self, access_token: str):
        """
        ref: https://zenn.dev/team_zenn/articles/117424abb5605b
        :param access_token:
        """
        if not isinstance(access_token, str):
            raise TypeError

        self.headers = {
            "Accept": "application/json",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + access_token
        }

    def show_database(self, database_id: str):
        try:
            result_list = []
            url = f'https://api.notion.com/v1/databases/{database_id}/query'
            response = requests.post(url, headers=self.headers)
            if 200 != response.status_code:
                raise
            if 'json' in response.headers.get('content-type'):
                self.result_dict = response.json()
                result = self.result_dict["results"]

                for item in result:
                    tmp_dict = {}
                    for key in NotionModel.column_name_list:  # todo: change to item['properties'].keys() ?
                        attribute = item['properties'][key]
                        match attribute['type']:
                            case 'number':
                                value = attribute['number']
                            case 'title':
                                if len(attribute['title']):
                                    value = attribute['title'][0]['text']['content']
                                else:
                                    value = None
                            case 'multi_select':
                                value = attribute['multi_select']
                            case _:
                                value = None
                                logger.info(f'{attribute.type=}')
                        tmp_dict[key] = value
                    result_list.append(tmp_dict)
            else:
                raise
        except Exception as e:
            logger.error(e)
        else:
            return result_list
