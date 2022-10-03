import logging

from py3pin.Pinterest import Pinterest

logger = logging.getLogger(__name__)


class PinterestService:
    """
    ref: https://github.com/bstoilov/py3-pinterest
    """

    image_info_list: list = None

    query_list: list = ['鞘師里保', '小田さくら', '佐藤優樹', 'モーニング娘。']

    def __init__(self):
        # self.pinterest = Pinterest(email='email',
        #                            password='password',
        #                            username='username',
        #                            cred_root='cred_root')
        try:
            self.pinterest = Pinterest()
        except Exception as e:
            pass

    def search(self, query: str, num_pins: int, scope: str = 'boards'):
        """
        ref: https://github.com/bstoilov/py3-pinterest/blob/master/examples.py#L146
        :param query:
        :param num_pins:
        :param scope:
        :return:
        """
        search_batch = self.pinterest.search(scope=scope, query=query, page_size=num_pins)
        """
        results = []
        while len(search_batch) > 0 and len(results) < num_pins:
            results += search_batch
            search_batch = self.pinterest.search(scope=scope, query=query)
        self.image_info_list: list = [item['image_cover_hd_url'] for item in results]
        """

        self.image_info_list: list = [item['image_cover_hd_url'] for item in search_batch]

    def _search(self, query: str):
        return self.pinterest.search(scope='boards', query=query)
