import os
import math
import shutil
import logging
import requests

import bs4
from PIL import Image

from .utils.convert_zipfile import ZipFile

logger = logging.getLogger(__name__)


class ImageService:
    IMAGE_EXTENSIONS: list = ['.jpg', '.png', '.jpeg']

    def __init__(self):
        pass

    def set_image(self, fp):
        self.image = Image.open(fp=fp)

    def is_exists_in_url(self, url: str) -> bool:
        _, ext = os.path.splitext(url)
        # return 0 != len([extension for extension in self.IMAGE_EXTENSIONS if extension in url])
        return ext in self.IMAGE_EXTENSIONS


class SearchImageService(ImageService):
    BASE_URL: str = ''

    def __init__(self):
        super(SearchImageService, self).__init__()

    def main(self, url: str = None):

        if url is None:
            url: str = self.BASE_URL

        res = requests.get(url)
        res.raise_for_status()
        soup = bs4.BeautifulSoup(res.text, "html.parser")

        cnt = 0

        # soup.find_all('a')
        # soup.find(class_='list ect-entry-card front-page-type-index')

        final_result: list = []

        for link in soup.find(id='list').find_all('a'):
            page_link: str = link.get("href")
            print(page_link)
            final_result.extend(self.get_img_link(url=page_link))
            if cnt == 5:
                break
            else:
                cnt += 1

    def get_img_link(self, url: str = None) -> list:
        if url is None:
            url = ''

        result_list: list = []

        # todo: analysis robots.txt

        res = requests.get(url)
        res.raise_for_status()
        soup = bs4.BeautifulSoup(res.text, "html.parser")

        for link in soup.find_all('img'):
            img_link: str = link.get("src")

            if img_link and self.is_exists_in_url(url=img_link):
                result_list.extend([img_link])
        return result_list


class DownloadImageService(ImageService):
    """
    ref: https://qiita.com/kazuooooo/items/37abb45c7806dbe31f14
    """
    API_PATH = "https://www.googleapis.com/customsearch/v1"

    PARAMS = {
        "cx": None,  # 検索エンジンID
        "key": None,  # APIキー
        "q": None,  # 検索ワード
        "searchType": "image",  # 検索タイプ
        "start": 1,  # 開始インデックス
        "num": 10  # 1回の検索における取得件数(デフォルトで10件)
    }
    LOOP = 10
    image_idx = 0

    query_list: list = ['鞘師里保', '小田さくら', '佐藤優樹', 'モーニング娘。']

    def __init__(self, cx: str, key: str):
        super(DownloadImageService, self).__init__()

        if 0 == len(cx) or 0 == len(key):
            raise ValueError
        self.PARAMS.update(cx=cx, key=key)

    def main(self, query: str = None):
        try:
            if 0 == len(query):
                raise ValueError

            self.PARAMS.update(q=query)
            file_dict: dict = self.get_images()
        except Exception as e:
            logger.error(e)
            raise e

    def get_images(self):
        try:
            logger.info(f'{self.PARAMS=}')
            res = requests.get(self.API_PATH, self.PARAMS).json()
            logger.info(f'{res=}')

            result_dict: dict = {}
            if 'error' in res and 429 == res['error']['code']:
                pass
            elif 'items' not in res:
                pass
            else:
                items_json = res["items"]

                index_num = 0
                logger.info(f'{len(items_json)=}')
                for item_json in items_json:
                    if self.is_exists_in_url(url=item_json['link']):
                        r = requests.get(item_json['link'], stream=True)
                        if r.status_code == 200:
                            filename: str = os.path.basename(item_json['link'])
                            result_dict[filename] = r.raw.read()
                            # self.save_image(filename=filename, obj=r.raw)  # todo: save_image flow
                    else:
                        r = requests.get(item_json['link'], stream=True)
                        if r.status_code == 200:
                            result_dict[f'{index_num}.jpg'] = r.raw.read()
                    index_num += 1
            return result_dict
        except Exception as e:
            logger.error(f'{self.PARAMS=}')
            raise e

    def download_images_as_zipfile(self, buffer, query: str, num: int = None):
        try:
            if 0 == len(query):
                raise ValueError
            self.PARAMS.update(q=query)
            if None in self.PARAMS.values():
                raise AttributeError

            result_dict: dict = {}
            num_index = math.floor(num / self.PARAMS['num'])

            for index in range(num_index):
                self.PARAMS.update(start=index + 1)
                result_dict = {**result_dict, **self.get_images()}
                logger.info(f'{index=}, {result_dict.keys()=}')

                if 0 == len(result_dict.keys()):
                    raise ValueError

            if 0 != num % self.PARAMS['num']:
                self.PARAMS.update(start=self.PARAMS['start'] + 1,
                                   num=num % self.PARAMS['num'])
                result_dict = {**result_dict, **self.get_images()}
                logger.info(f'{result_dict.keys()=}')

            zipfile = ZipFile()
            zipfile.file_data_dict = result_dict
            return zipfile.main(buffer=buffer)

        except Exception as e:
            logger.error(e)
            raise e

    def save_image(self, filename: str, obj):
        dir_name: str = os.path.join('images', self.PARAMS['q'])
        os.makedirs(dir_name, exist_ok=True)

        path: str = os.path.join(dir_name, filename)

        with open(path, 'wb') as f:
            obj.decode_content = True
            shutil.copyfileobj(obj, f)
