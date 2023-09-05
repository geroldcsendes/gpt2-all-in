from argparse import ArgumentParser
import json
from pprint import pprint
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
from gpt2_ai.train.trainer import Trainer
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

    sample = next(iter(loader))["input_ids"]
    print("Sample:\n", sample)
    print(sample.shape)
    out = model(sample)

    print(out)