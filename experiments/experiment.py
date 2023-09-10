# %%
from gpt2_ai.train import dataset
from gpt2_ai.train.config import GPT2Config
from transformers import GPT2Tokenizer, GPT2Tokenizer

# %%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
cfg = GPT2Config()
ds = dataset.Dev(tokenizer=tokenizer, stream=False,
                 model_config=cfg)


# %%
ds.get_dataset()

# %%
ds.model_config
# %%

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
# %%
pile = dataset.Pile(tokenizer=tokenizer, stream=False)
# %%
pile.get_dataset()
# %%
GPT2_params = 127_000_000
float32_byte = 4

gpt2_params_in_gb = GPT2_params * float32_byte / 1e9
print(gpt2_params_in_gb)
# %%