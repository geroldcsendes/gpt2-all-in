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


class RewardModel(t.nn.Module):
    def __init__(self, base_model: GPT2Model):
        super().__init__()
        self.base = base_model
        self.reward_scalar = t.nn.Linear(in_features=base_model.config.n_embd, out_features=1)

    def forward(self, **kwargs):
        base_out = self.base(**kwargs, output_attentions=False, use_cache=False)
        # only take last token's hidden state
        base_out = base_out.last_hidden_state[:,-1,:]
        out = self.reward_scalar(base_out)

        return out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        base_model = GPT2Model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(base_model)

        # Load the additional linear layer
        model.linear = t.nn.Linear(in_features=base_model.config.hidden_size, out_features=1)
        model.linear.load_state_dict(
            t.load(os.path.join(pretrained_model_name_or_path, "pytorch_model_lin.bin"),
                   map_location='cpu'))

        return model

    def save_pretrained(self, save_directory):
        self.base.save_pretrained(save_directory)
        t.save(
            self.reward_scalar.state_dict(),
            os.path.join(save_directory, "pytorch_model_lin.bin"))


if __name__ == '__main__':

    # reward_base = GPT2Model.from_pretrained('gpt2')
    # reward_model = RewardModel(base_model=reward_base)

    # print(reward_model)

    # save model
    # reward_model.save_pretrained('/home/gerold/projects/gpt2-all-in/upload/reward')


    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = tokenizer.eos_token

    # load model
    #path = '/home/gerold/projects/gpt2-all-in/upload/reward'
    path = '/home/gerold/projects/RLHF_art/RM/11-12/rm-prod-23-11-12-20:20:13'

    reward_model = RewardModel.from_pretrained(path)
    print(reward_model)

    # ckp = t.load(os.path.join(path, "pytorch_model_lin.bin"),
    #                map_location='cpu')  # ['reward_scalar']
    # print(ckp)

    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('/home/gerold/projects/RLHF_art/SFT-1108-1epoch/logs/sft-hh-rlhf-23-11-08-19:33:25')
    # print(model)
    # model = GPT2LMHeadModel.from_pretrained('geroldcsendes/sft-hh-rlhf')
    # print(model)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = tokenizer.eos_token

    # model.save_pretrained('/home/gerold/projects/gpt2-all-in/upload')
    # tokenizer.save_pretrained('/home/gerold/projects/gpt2-all-in/upload')

