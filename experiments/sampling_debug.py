from argparse import ArgumentParser
import json
from pprint import pprint
from tqdm import tqdm

import os.path as osp
import shutil
from typing import Tuple

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch as t
from torch.utils.data import DataLoader
import numpy as np
from datasets import Dataset

from gpt2_ai.train.model import GPT2
from gpt2_ai.train.sample import TransformerSampler
# from gpt2_ai.train.trainer import Trainer
from gpt2_ai.train.config import GPT2Config, TrainerConfig


if __name__ == "__main__":

    device = "cuda" if t.cuda.is_available() else "cpu"

    # custom model
    dscr_name = '/home/gerold/projects/gpt2-all-in/logs/vibrant_euclid/config.json'
    dscr = json.load(open(dscr_name, 'r'))

    # parse model config
    conf_model = GPT2Config.model_validate(dscr['model'])
    # parse trainer config
    dscr['trainer']['dataset'] = "debug"
    conf_trainer = TrainerConfig.model_validate(dscr['trainer'])

    model = GPT2(config=conf_model, config_trainer=conf_trainer)
    model.to(device)

    custom_model_ckp = '/home/gerold/projects/gpt2-all-in/logs/vibrant_euclid/ckpt/step-6000.pt'
    model.load_state_dict(t.load(custom_model_ckp, map_location=t.device('cpu')))

    model.count_params()

    # # Huggingface
    # model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # prompt = "Hello, my name is "

    # sampler = TransformerSampler(model, tokenizer)

    # out = sampler.sample(prompt, verbose=False, max_tokens_generated=40)

    # print('Generated text: ', out)

