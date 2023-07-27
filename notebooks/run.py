# %%
from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch as t
from torch.utils.data import DataLoader, Dataset

from gpt2.model import GPT2, CausalSelfAttention, Block
from gpt2.trainer import Trainer
from gpt2.config import GPT2Config, TrainerConfig

if __name__ == "__main__":
    # %%
    conf = GPT2Config(n_layer=4, d_model=64, n_ctx=24, n_head=8)
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # %%
    dataset_name = "stas/openwebtext-10k"
    name = 'data/' + dataset_name.split('/')[-1]
    ds = load_dataset(dataset_name, split='train')

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # %%
    # test fwd pass
    model = GPT2(conf)
    model._count_params()

    model.to(device)

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    sample = next(iter(loader))
    in_x = tokenizer(
        sample['text'], return_tensors='pt',
        return_attention_mask=False,
        max_length=conf.n_ctx,
        padding='max_length', truncation=True)
    in_x.to(device)

    logits = model(in_x['input_ids'])
    print("Logits shape", logits.shape)

    # %%
    # train model
    model = GPT2(conf)
    model.to(device)

    trainer_conf = TrainerConfig(ckpt_path='ckpt', log_path='logs', batch_size=512, n_epochs=50)
    loader = DataLoader(ds, trainer_conf.batch_size, shuffle=True)
    trainer = Trainer(trainer_conf, model, loader, loader, tokenizer)

    trainer.train()
