import datasets
from transformers import AutoTokenizer
# from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm, trange
import torch as t
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, TensorDataset

from torch.utils.tensorboard import SummaryWriter
# from accelerate import Accelerator
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, Trainer, TrainingArguments
from datetime import datetime
import os.path as osp
import shutil
from argparse import ArgumentParser

import os

def train_step(model, batch, tokenizer, criterion):
    model.train()

    logits = model(**batch).logits[:,:-1,:]
    logits = logits.reshape(-1, tokenizer.vocab_size)  # [batch_size*seq_len, vocab_size]
    target = batch['input_ids'][:,1:].reshape(-1)  # [batch_size*seq_len]

    loss = criterion(logits, target)

    return loss


if __name__ == '__main__':

    BS = 2
    DEV = True

    if DEV:
        model_str = 'gpt2'
    else:
        model_str = 'gpt2-medium'

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    # get max vram

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    ds = t.randint(0, tokenizer.vocab_size, (1000, 1000))

    dataset = TensorDataset(ds)
    dataloader = DataLoader(dataset, batch_size=BS)

    model = GPT2LMHeadModel.from_pretrained(model_str).to(device)
    criterion = t.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    optimizer = Adam(model.parameters(), 6e-5)
    print(ds)
    print(dataset)

    for e in range(100):
        for cnt, batch in enumerate(dataloader):

            batch = {"input_ids": batch[0].to(device)}
            loss = train_step(model, batch, tokenizer, criterion)
            loss.backward()

            print(f'Batch: {cnt}, Loss: {loss.item()}')