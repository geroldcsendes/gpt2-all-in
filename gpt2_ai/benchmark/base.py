from abc import ABC, abstractmethod
from typing import Union, Literal

from transformers import PreTrainedModel, PreTrainedTokenizer

from gpt2_ai.model import GPT2


class BaseBenchmark(ABC):
    def __init__(self, name: str, model: Union[PreTrainedModel, GPT2],
                 tokenizer: PreTrainedTokenizer, device: Literal['cpu', 'cuda']):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def dataset(self):
        """
        Parse the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> float: 
        """
        Run the benchmark and return the score.
        """
        raise NotImplementedError
    
    @abstractmethod
    def debug_examples(self):
        """
        Print a few examples of the model's predictions and the ground truth.
        """
        raise NotImplementedError