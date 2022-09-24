import logging

from keybert import KeyBERT
from sentence_transformers import models, SentenceTransformer

logger = logging.getLogger(__name__)


class KeyBERTService:
    """
    学習済みのBERTに新たにTokenizer(キーワード)を追加。
    Sentence-Transformersのモデルに学習済みモデルを読み込む。
    Sentence-Transformersのモデルにキーワードを追加する？
    「キーワードを追加したSentence-Transformersのモデル」をKeyBERTのEmbedding Modelsとして、適用させる。
    """

    stop_words_list: list = [
        ',',
        '.',
        '。',
        '、',
        '_',
        'ます',
        'ました',
    ]

    word_embedding_model_list: list = [
        'rinna/japanese-roberta-base',
        'bert-base-uncased',
        # 'all-MiniLM-L6-v2',
    ]

    example_value_dict: dict = {
        'rinna/japanese-roberta-base': 'おはようございます。本日は晴天なり。'
                                       'こんにちは、ランチタイムでございます。'
                                       'こんばんは。帰路に就き始めました。',
        'bert-base-uncased': 'hogefoofoofoofoohogehogehogehoge',
        # 'all-MiniLM-L6-v2': 'This is a pen.',
    }

    example_add_tokens_dict = {
        'rinna/japanese-roberta-base': '例) おはようございます 本日 晴天 こんにちは ランチタイム こんばんは 帰路',
        'bert-base-uncased': 'Example) hoge foo',
        # 'all-MiniLM-L6-v2': 'Example) This is a pen.',
    }

    def get_example_value(self, model_name: str):
        if model_name not in self.word_embedding_model_list:
            raise ValueError
        return self.example_value_dict[model_name]

    def get_example_add_tokens(self, model_name: str):
        if model_name not in self.word_embedding_model_list:
            raise ValueError
        return self.example_add_tokens_dict[model_name]

    def main(self, model_name: str, payload: str, add_tokens: str = None, top_n: int = 5):
        """

        :param model_name:
        :param payload:
        :param add_tokens: example) 'hoge foo bar'
        :param top_n:
        :return:
        """
        if model_name not in self.word_embedding_model_list:
            raise ValueError
        if not isinstance(payload, str):
            raise ValueError
        if add_tokens is not None and not isinstance(add_tokens, str):
            raise TypeError

        # set up word_embedding_model
        # ref: https://www.sbert.net/docs/training/overview.html#adding-special-tokens
        word_embedding_model = models.Transformer(model_name_or_path=model_name)
        tokens = ["[DOC]", "[QRY]"]
        if add_tokens is not None:
            add_tokens_list: list = add_tokens.split()
            tokens = tokens + add_tokens_list
        logger.info(f'{tokens=}')
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # add base model to keybert
        # ref: https://github.com/MaartenGr/KeyBERT#25-embedding-models
        try:
            kw_model = KeyBERT(model=sentence_model)
        except Exception as e:
            logger.error(e)
            kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')

        tokens = word_embedding_model.tokenizer.tokenize(payload)
        tokens = [token for token in tokens if token not in self.stop_words_list]
        tokens = ' '.join(tokens)
        logger.info(f'{tokens=}')
        result = kw_model.extract_keywords(tokens, top_n=top_n, keyphrase_ngram_range=(1, 1), stop_words='english')
        logger.info(f'{result=}')
        return result

    def base_extract_keywords(self, payload: str, top_n: int = 5):
        """
        ref: https://crieit.net/posts/KeyBERT
        :return:
        """
        import MeCab
        # MeCabで分かち書き
        words = MeCab.Tagger("-Owakati").parse(payload)
        model = KeyBERT('distilbert-base-nli-mean-tokens')
        result = model.extract_keywords(words, top_n=top_n, keyphrase_ngram_range=(1, 1))
        return result
