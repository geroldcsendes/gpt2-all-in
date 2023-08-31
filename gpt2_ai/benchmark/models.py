from enum import Enum

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM

from gpt2_ai.benchmark.base import BaseModel


class ModelType(str, Enum):
    """Enum class for model types."""

    # Model types
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    GPT2_XL = "gpt2-xl"

    GPT_NEO_SMALL = "gpt_neo_small"


class GPT2(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.n_ctx = self.model.config.n_ctx

        # other configs
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.generation_config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token


class GPTNeo(BaseModel):
    def __init__(self, model: ModelType):
        super().__init__()
        self._select_model_tknizer(model)
        self.n_ctx = self.model.config.window_size

        # other configs
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.generation_config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def _select_model_tknizer(self, model: ModelType):
        if model == ModelType.GPT_NEO_SMALL:
            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")    
        # todo: add other models
        else:
            raise ValueError(f"Unknown model type: {model}")
