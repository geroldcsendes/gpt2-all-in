# %%
from argparse import ArgumentParser
import json
from pprint import pprint
import os.path as osp
import shutil
from typing import Tuple

from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch as t
from torch.utils.data import DataLoader, Dataset

from gpt2_ai.model import GPT2
from gpt2_ai.trainer import Trainer
from gpt2_ai.config import GPT2Config, TrainerConfig


def setup_loader(**kwargs) -> DataLoader:

    dev = kwargs['dev']

    seed, buffer_size = 42, 10_000
    
    streaming = True
    dataset_name = 'Skylion007/openwebtext'
    if dev:
        streaming = False
        dataset_name = 'stas/openwebtext-10k'

    ds = load_dataset(dataset_name, split='train', streaming=streaming)

    def encode(example):
        return tokenizer(
            example['text'], truncation=True, padding='max_length',
            max_length=kwargs['conf_model'].n_ctx,
            return_tensors='np', return_attention_mask=False)
    
    ds = ds.map(
        encode, batched=True, remove_columns=['text'])
    
    ds = ds.with_format("torch")

    if dev:
        loader = DataLoader(ds, batch_size=conf_trainer.batch_size,
                            shuffle=True, num_workers=conf_trainer.n_workers)
    else:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        loader = DataLoader(ds, batch_size=conf_trainer.batch_size,
                            num_workers=conf_trainer.n_workers)

    return loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='cpu.json')

    args = parser.parse_args()
    dscr_name = osp.join('configs', args.config)
    dscr = json.load(open(dscr_name, 'r'))
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # parse dataset
    if dscr['dataset'] == 'dev':
        dataset_name = 'stas/openwebtext-10k'
        dev = True
    elif dscr['dataset'] == 'full':
        dataset_name = 'Skylion007/openwebtext'
        dev = False
    else:
        raise ValueError("Unknown dataset name")        

    # parse model config
    conf_model = GPT2Config.model_validate(dscr['model'])
    # parse trainer config
    dscr['trainer']['dataset'] = dataset_name
    conf_trainer = TrainerConfig.model_validate(dscr['trainer'])    

    device = conf_trainer.device
    print("\nUsing device:", device)

    print('\nUsing hyperparameters:')
    pprint(conf_model.model_dump(), indent=2)

    print('\nUsing training settings:')
    pprint(conf_trainer.model_dump(), indent=2)

    loader = setup_loader(
        dev=dev,
        conf_model=conf_model,
        conf_trainer=conf_trainer,
        tokenizer=tokenizer
        )

    model = GPT2(conf_model)
    model.to(device)
    model.count_params()  # print the number of parameters

    trainer = Trainer(config=conf_trainer,
                      model=model,
                      train_loader=loader,
                      tokenizer=tokenizer)
    
    dscr_output_dir = osp.join(conf_trainer.log_path, trainer.run_name, 'config.json')
    shutil.copy(dscr_name, dscr_output_dir)

    trainer.train()
