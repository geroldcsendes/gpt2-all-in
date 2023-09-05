from argparse import ArgumentParser
import json
from pprint import pprint
from tqdm import tqdm

import os.path as osp
import shutil
from typing import Tuple

from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch as t
from torch.utils.data import DataLoader
import numpy as np
from datasets import Dataset

from gpt2_ai.train.model import GPT2
# from gpt2_ai.train.trainer import Trainer
from gpt2_ai.train.config import GPT2Config, TrainerConfig


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='cpu.json')

    args = parser.parse_args()
    dscr_name = osp.join('configs', args.config)
    dscr = json.load(open(dscr_name, 'r'))

    # parse model config
    conf_model = GPT2Config.model_validate(dscr['model'])
    # parse trainer config
    dscr['trainer']['dataset'] = "debug"
    conf_trainer = TrainerConfig.model_validate(dscr['trainer'])

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset_size = 100
    seq_len = conf_model.n_ctx

    dummy_data = {
        "input_ids": np.random.randint(100, 50_000, (dataset_size, seq_len))}

    ds = Dataset.from_dict(dummy_data)
    ds.set_format("pt")

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    device = conf_trainer.device
    print("\nUsing device:", device)

    print('\nUsing hyperparameters:')
    pprint(conf_model.model_dump(), indent=2)

    print('\nUsing training settings:')
    pprint(conf_trainer.model_dump(), indent=2)

    model = GPT2(conf_model)
    model.to(device)
    model.count_params()  # print the number of parameters

    optimizer = t.optim.Adam(model.parameters(), lr=conf_trainer.lr)
    criterion = t.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


    EPOCHS = 100
    model.train()
    for epoch in range(EPOCHS):
        for batch in loader:
            batch = batch["input_ids"]
            batch = batch.to(device)

            logits = model(batch)[:,:-1,:]

            logits = logits.reshape(-1,tokenizer.vocab_size)  # [batch_size*seq_len, vocab_size]
            target = batch[:,1:].reshape(-1)  # [batch_size*seq_len]

            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()


    # sample = next(iter(loader))["input_ids"]
    # sample = sample.to(device)

    # print("Sample:\n", sample)
    # print(sample.shape)
    # out = model(sample)

    # print(out)