import json
import random
import logging
import requests

import torch
import transformers
from transformers import AutoTokenizer
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import RobertaForMaskedLM

logger = logging.getLogger(__name__)

SEED: int = 42
transformers.utils.move_cache()
transformers.trainer.set_seed(seed=SEED)


class HuggingFaceService:
    API_URL: str = None

    # Task
    task_list = [
        'Text Generation',
        'Fill-Mask'
    ]

    # Model Per Task
    text_generation_model_id_list: list = [
        'gpt2',  # https://huggingface.co/gpt2
    ]
    fill_mask_model_id_list: list = [
        'roberta-base',  # https://huggingface.co/roberta-base
        'bert-base-uncased',  # https://huggingface.co/bert-base-uncased
    ]
    all_model_id_list = text_generation_model_id_list + fill_mask_model_id_list

    # Example Value
    example_value_dict = {
        'gpt2': 'My name is Thomas and my main',
        'roberta-base': 'The goal of life is <mask>.',
        'bert-base-uncased': 'The goal of life is [MASK].'
    }

    def __init__(self, access_token: str):
        if not isinstance(access_token, str):
            raise TypeError
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def get_example_value(self, model_name):
        try:
            if not isinstance(model_name, str):
                raise TypeError
            if model_name not in self.all_model_id_list:
                raise ValueError
            return self.example_value_dict[model_name]
        except KeyError as e:
            logger.error(e)
            return ''

    def set_api_url(self, model_name: str):
        try:
            if not isinstance(model_name, str):
                raise TypeError
            if model_name not in self.all_model_id_list:
                raise ValueError
            self.API_URL = f'https://api-inference.huggingface.co/models/{model_name}'
        except Exception as e:
            raise e

    def query(self, payload: str, model_name: str):
        """
        ref: https://note.com/npaka/n/n3d439c8b0930
        :return:
        """
        try:
            if self.API_URL is None:
                self.set_api_url(model_name=model_name)
            data = json.dumps(payload)
            response = requests.request("POST", self.API_URL, headers=self.headers, data=data)
            result = json.loads(response.content.decode("utf-8"))
        except Exception as e:
            logger.error(e)
        else:
            self.API_URL = None
            return result


class HuggingFaceBuiltInService:
    """
    ref: https://note.com/npaka/n/n96dde45fdf8d

    todo: add
        'rinna/japanese-stable-diffusion', # https://huggingface.co/rinna/japanese-stable-diffusion
    """
    tokenizer: T5Tokenizer.from_pretrained = None
    model: AutoModelForCausalLM = None

    # Task
    task_list = [
        'Text Generation',
        'Fill-Mask'
    ]

    # Model Per Task
    text_generation_model_id_list = [
        'rinna/japanese-gpt2-medium',  # https://huggingface.co/rinna/japanese-gpt2-medium
    ]
    fill_mask_model_id_list = [
        'rinna/japanese-roberta-base',  # https://huggingface.co/rinna/japanese-roberta-base
    ]
    all_model_id_list = text_generation_model_id_list + fill_mask_model_id_list

    example_value_dict = {
        'rinna/japanese-gpt2-medium': '生命、宇宙、そして万物についての究極の疑問の答えは',
        'rinna/japanese-roberta-base': '[CLS]4年に1度[MASK]は開かれる。'
    }

    def get_example_value(self, model_name: str):
        try:
            if not isinstance(model_name, str):
                raise TypeError
            if model_name not in self.all_model_id_list:
                raise ValueError
            return self.example_value_dict[model_name]
        except KeyError as e:
            logger.error(e)
            return ''

    def set_up(self, model_name: str):
        if not isinstance(model_name, str):
            raise TypeError
        if model_name not in model_name:
            raise ValueError
        self._set_tokenizer(model_name=model_name)
        self._set_model(model_name=model_name)
        if self.tokenizer is None or self.model is None:
            raise ValueError

    def _set_tokenizer(self, model_name: str):
        try:
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        except Exception as e:
            logger.error(e)
            raise e

    def _set_model(self, model_name: str):
        if model_name in self.text_generation_model_id_list:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_name in self.fill_mask_model_id_list:
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
        else:
            raise ValueError

    def inference(self, model_name: str, payload: str):
        if self.tokenizer is None and self.model is None:
            self.set_up(model_name=model_name)
        elif self.tokenizer is None or self.model is None:
            raise AttributeError

        if model_name in self.text_generation_model_id_list:
            input = self.tokenizer.encode(payload, return_tensors="pt")
            output = self.model.generate(input, do_sample=True, max_length=30, num_return_sequences=3)
            logger.info(self.tokenizer.batch_decode(output))
            result = self.tokenizer.batch_decode(output)
        elif model_name in self.fill_mask_model_id_list:
            result = self._fill_mask_inference(payload=payload)
        else:
            raise ValueError

        self.tokenizer = self.model = None
        return result

    def _fill_mask_inference(self, payload: str):
        """
        ref: https://huggingface.co/rinna/japanese-roberta-base
        :param payload:
        :return:
        """
        if not isinstance(payload, str):
            raise TypeError

        tokens = self.tokenizer.tokenize(payload)
        if '[CLS]' not in tokens:
            tokens = ['[CLS]'] + tokens
        if '[MASK]' not in tokens:
            _mask_index = random.randint(1, len(tokens) - 1)
            tokens[_mask_index] = self.tokenizer.mask_token

        masked_idx = tokens.index('[MASK]')

        # convert to ids
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        logger.info(token_ids)  # output: [4, 1602, 44, 24, 368, 6, 11, 21583, 8]

        # convert to tensor
        token_tensor = torch.LongTensor([token_ids])

        # provide position ids explicitly
        position_ids = list(range(0, token_tensor.size(1)))
        logger.info(position_ids)  # output: [0, 1, 2, 3, 4, 5, 6, 7, 8]
        position_id_tensor = torch.LongTensor([position_ids])

        # get the top 10 predictions of the masked token
        with torch.no_grad():
            outputs = self.model(input_ids=token_tensor, position_ids=position_id_tensor)
            predictions = outputs[0][0, masked_idx].topk(10)

        result = []
        for i, index_t in enumerate(predictions.indices):
            index = index_t.item()
            token = self.tokenizer.convert_ids_to_tokens([index])[0]
            result.append({'index': i, 'token': token})

        return result
