from abc import ABC, abstractmethod
from functools import wraps
from random import sample
from typing import Union, Literal

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets.arrow_dataset import Dataset

from gpt2_ai.model import GPT2


def dev_mode_sample(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        
        dataset = func(self, *args, **kwargs)
        if self.dev_mode:
            # Modify behavior for dev mode
            # For example, return a small sample of the dataset
            rand_indices = sample(range(len(dataset)), k=100)
            if type(dataset) == list:
                dataset = [element for cnt, element in enumerate(dataset) if cnt in rand_indices]
            elif type(dataset) == Dataset:
                dataset = dataset.select(rand_indices)
            else:
                raise ValueError(f"Unknown dataset type: {type(dataset)}")
        return dataset
    
    return wrapper


class BaseDataset(ABC):
    def __init__(self, dev_mode: bool = False):
        self.dev_mode = dev_mode

    @abstractmethod
    @dev_mode_sample
    def get_dataset(self):
        """
        Parse the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, batch_size: int):
        """
        Return a batch of data.
        """
        raise NotImplementedError


class BaseBenchmark(ABC):
    def __init__(self, model: Union[PreTrainedModel, GPT2],
                 tokenizer: PreTrainedTokenizer, device: Literal['cpu', 'cuda'],
                 dataset: BaseDataset):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = dataset
        self.debug_sample_num = 5  # Number of samples to show when debugging

    @abstractmethod
    def run(self, **kwargs) -> float: 
        """
        Run the benchmark and return the score.
        """
        raise NotImplementedError
    
    @abstractmethod
    def debug_examples(self, **kwargs):
        """
        Print a few examples of the model's predictions and the ground truth.
        """
        raise NotImplementedError
    

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.n_ctx = None
