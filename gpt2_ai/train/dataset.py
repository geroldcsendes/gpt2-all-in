"""
Datasets available for training.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union
from transformers import GPT2Tokenizer

from datasets import load_dataset

from gpt2_ai.train.config import GPT2Config


class DatasetType(str, Enum):
    OPENWEBTEXT = "openwebtext"
    WIKITEXT2 = "wikitext2"
    WIKITEXT103 = "wikitext103"
    PILE = "pile"
    DEV = "dev"


class BaseDataset(ABC):
    """
    Abstract class for training datasets
    """
    def __init__(self, tokenizer: GPT2Tokenizer, stream: bool,
                 model_config: GPT2Config = None):
        self.stream = stream
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.huggingface_repo = None
        self.dataset = None

    @abstractmethod
    def get_dataset(self):
        """
        Parse the dataset.
        """
        raise NotImplementedError

    # @abstractmethod
    # def encode(self, **kwargs):
    #     """
    #     Encode the dataset.
    #     """
    #     raise NotImplementedError

    # @abstractmethod
    # def map_dataset(self):
    #     """
    #     Map the dataset.
    #     """
    #     raise NotImplementedError

    # @abstractmethod
    # def get_batch(self, batch_size: int):
    #     """
    #     Return a batch of data.
    #     """
    #     raise NotImplementedError


class OpenWebText(BaseDataset):
    def __init__(self, stream=True, **kwargs):
        super().__init__(stream=stream, **kwargs)
        self.huggingface_repo = 'Skylion007/openwebtext'
        self.tokenizer = kwargs['tokenizer']
        self.stream = stream

    def get_dataset(self):
        # todo: dedicate a separate function to encode the dataset
        def encode(example):
            return self.tokenizer(
                example['text'], truncation=True, padding='max_length',
                max_length=self.model_config.n_ctx,
                return_tensors='np', return_attention_mask=False)

        ds = load_dataset(
            self.hugginface_repo,
            split='train', streaming=self.stream)

        print("Encoding dataset...")
        # TODO: chunk examples to n_ctx length. Use smaller batch_size and multiprocessing
        ds = ds.map(encode, batched=True, remove_columns=['text'])

        ds = ds.with_format("torch")

        self.dataset = ds


class Dev(BaseDataset):
    def __init__(self, stream=False, **kwargs):
        super().__init__(stream=stream, **kwargs)
        self.huggingface_repo = 'stas/openwebtext-10k'
        self.tokenizer = kwargs['tokenizer']
        self.stream = stream

    def get_dataset(self):

        # todo: dedicate a separate function to encode the dataset
        def encode(example):
            return self.tokenizer(
                example['text'], truncation=True, padding='max_length',
                max_length=self.model_config.n_ctx,
                return_tensors='np', return_attention_mask=False)

        ds = load_dataset(
            self.huggingface_repo,
            split='train', streaming=self.stream)

        print("Encoding dataset...")

        ds = ds.map(encode, batched=True, remove_columns=['text'])

        ds = ds.with_format("torch")

        self.dataset = ds


class Pile(BaseDataset):
    """
    The Pile dataset as described in: https://pile.eleuther.ai/
    DOES NOT WORK AT THE MOMENT: the download link is broket.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hugginface_repo = 'EleutherAI/pile'
        self.tokenizer = kwargs['tokenizer']

    def get_dataset(self):
        raise NotImplementedError
        # def encode(example):
        #     return self.tokenizer(
        #         example['text'], truncation=True, padding='max_length',
        #         max_length=self.model_config.n_ctx,
        #         return_tensors='np', return_attention_mask=False)

        # ds = load_dataset(
        #     self.hugginface_repo,
        #     split='train', streaming=self.stream)

        # print("Encoding dataset...")

        # ds = ds.map(encode, batched=True, remove_columns=['text'])

        # ds = ds.with_format("torch")

        # self.dataset = ds


def get_dataset(ds_name: str) -> BaseDataset:
    """
    Factory function for datasets.
    """

    if ds_name == DatasetType.OPENWEBTEXT:
        return OpenWebText
    elif ds_name == DatasetType.PILE:
        return Pile
    elif ds_name == DatasetType.DEV:
        return Dev
    else:
        raise ValueError("Unknown dataset name: ", ds_name)