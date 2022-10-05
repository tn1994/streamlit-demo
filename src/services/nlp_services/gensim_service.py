import random

import gensim
import gensim.downloader as api
from gensim.models import FastText
from gensim.models import word2vec


class GensimService:
    """
    ref:
    https://qiita.com/MuAuan/items/d46985223cad76764b9d
    https://huggingface.co/Gensim/glove-twitter-25
    """
    random_positive_list: list = [
        '東京', 'フランス', '北京'
        # 'Tokyo', 'France', 'America'
    ]

    random_negative_list: list = [
        'すごい', '悲しい', '面白い'
        # 'Cool', 'Amazing', 'OMG'
    ]

    most_similar_list: list = []

    def __init__(self):
        # self.model = FastText.load_fasttext_format('model300.bin')
        self._setup_model()
        self.random_positive_idx: int = self._get_positive_idx()
        self.random_negative_idx: int = self._get_negative_idx()

    def _setup_model(self):
        filename = ''
        # sentences = gensim.models.word2vec.Text8Corpus(filename)
        # self.model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

        # coupus = word2vec.Text8Corpus(filename)
        # self.model = word2vec.Word2Vec(coupus, size=200, window=5, min_count=5)

        # self.model = api.load("glove-twitter-25", from_hf=True)
        self.model = api.load("glove-twitter-25")
        # model.most_similar(positive=['fruit', 'flower'], topn=1)

    def main(self):
        self.most_similar_list: list = self.model.most_similar(
            positive=[
                self.random_positive_list[self.random_positive_idx]
            ],
            negative=[
                self.random_negative_list[self.random_negative_idx]
            ])

    def _get_positive_idx(self):
        return random.randint(0, len(self.random_negative_list))

    def _get_negative_idx(self):
        return random.randint(0, len(self.random_negative_list))

    def is_correct_answer(self, word: str) -> bool:
        if word not in self.random_negative_list:
            return False
        elif word == self.random_negative_list[self.random_negative_idx]:
            return True
        elif word != self.random_negative_list[self.random_negative_idx]:
            return False
        else:
            raise
