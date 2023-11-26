import datasets
from transformers import AutoTokenizer
# from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm, trange
import torch as t
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter
# from accelerate import Accelerator
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, Trainer, TrainingArguments
from datetime import datetime
import os.path as osp
import shutil
from argparse import ArgumentParser

import os

from transformers import get_constant_schedule_with_warmup

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='dev mode', default=False)

    return parser.parse_args()


def get_loader(dev, tokenizer, bs=4):

    dataset = datasets.load_dataset('anthropic/hh-rlhf')['train']
    # take the otherhalf of the dataset: sft contains range(len(dataset)//2)
    indices = range(len(dataset)//2, len(dataset))
    dataset = dataset.shuffle(seed=1).select(indices)
    print('dataset size:', len(dataset))

    # split into train and val
    traintest = dataset.train_test_split(test_size=0.1, seed=1)
    train = traintest['train']
    val = traintest['test']
    del dataset, traintest

    if dev:
        train = train.select(range(100))
        val = val.select(range(20))

    print('train size:', len(train))
    print('val size:', len(val))

    def _encode_batch(batch):
        # only tkae chosens for sft
        chosen = [element['chosen'] for element in batch]
        rej = [element['rejected'] for element in batch]

        chosen = tokenizer(
            chosen,  padding=True, truncation=True,
            return_tensors='pt', return_attention_mask=True)

        rej = tokenizer(rej,  padding=True, truncation=True,
                      return_tensors='pt', return_attention_mask=True)

        # get the last non-pad token idx
        last_idx_chosen = chosen['attention_mask'].sum(dim=1) - 1
        last_idx_rej = rej['attention_mask'].sum(dim=1) - 1
        last_idx_chosen = last_idx_chosen.to(t.long)
        last_idx_rej = last_idx_rej.to(t.long)

        del chosen['attention_mask'], rej['attention_mask']

        enc_batch = dict(
            chosen=chosen,
            rejected=rej
        )

        last_nonpad_idx = dict(
            chosen=last_idx_chosen,
            rejected=last_idx_rej
        )

        return enc_batch, last_nonpad_idx

    train_loader = t.utils.data.DataLoader(
        train, batch_size=bs, shuffle=True, collate_fn=_encode_batch,
        num_workers=4, pin_memory=True)
    val_loader = t.utils.data.DataLoader(
        val, batch_size=2*bs, shuffle=True, collate_fn=_encode_batch,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader


class RewardModel(t.nn.Module):
    def __init__(self, base_model: GPT2Model):
        super().__init__()
        self.base = base_model
        self.reward_scalar = t.nn.Linear(in_features=base_model.config.n_embd, out_features=1)

    def forward(self, last_nonpad_idx, **kwargs):

        base_out = self.base(**kwargs, output_attentions=False, use_cache=False)

        # only take last token's hidden state
        bs = base_out.last_hidden_state.size(0)
        base_out= (
            base_out.last_hidden_state[t.arange(bs),last_nonpad_idx,:])

        out = self.reward_scalar(base_out)

        return out

    def forward_value(self, **kwargs):

        base_out = self.base(**kwargs, output_attentions=False, use_cache=False)

        out = self.reward_scalar(base_out.last_hidden_state)

        return out

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, device,
        pretrained_lin_path,
        *model_args, **kwargs):
        base_model = GPT2Model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(base_model)

        # Load the additional linear layer
        model.linear = t.nn.Linear(in_features=base_model.config.hidden_size, out_features=1)
        model.linear.load_state_dict(
            t.load(pretrained_lin_path,
                   map_location=device))

        return model

    def save_pretrained(self, save_directory):
        self.base.save_pretrained(save_directory)
        t.save(
            self.reward_scalar.state_dict(),
            os.path.join(save_directory, "pytorch_model_lin.bin"))


def train_step(model, batch, device, tokenizer=None, criterion=None):
    model.train()

    chosen_input = batch[0]['chosen'].to(device)
    chosen_last_idx = batch[1]['chosen'].to(device)
    rejected_input = batch[0]['rejected'].to(device)
    rejected_last_idx = batch[1]['rejected'].to(device)

    chosen_out = model(chosen_last_idx, **chosen_input)
    rejected_out = model(rejected_last_idx, **rejected_input)

    y_hat = t.sigmoid(chosen_out - rejected_out)

    loss = - t.log(y_hat)
    # clip for safety
    loss = t.clip(loss, min=0., max=20.)
    loss = loss.mean()

    return loss

@t.inference_mode()
def valid_step(model: RewardModel, batch, device, tokenizer=None, criterion=None):
    model.eval()

    chosen_input = batch[0]['chosen'].to(device)
    chosen_last_idx = batch[1]['chosen'].to(device)
    rejected_input = batch[0]['rejected'].to(device)
    rejected_last_idx = batch[1]['rejected'].to(device)

    chosen_out = model(chosen_last_idx, **chosen_input)
    rejected_out = model(rejected_last_idx, **rejected_input)

    y_hat = t.sigmoid(chosen_out - rejected_out)

    loss = - t.log(y_hat)
    # clip for safety
    loss = t.clip(loss, min=0., max=20.)
    loss = loss.mean()

    return loss


def main():

    args = parse_args()

    if args.dev:
        BS = 2
    else:
        BS = 16

    RUN_NAME = 'RM/rm-dev'

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    # get max vram

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    out = tokenizer(
            'I want to have a',  padding=True, truncation=True,
            return_tensors='pt', return_attention_mask=True)

    print(out)

    train_loader, val_loader = get_loader(args.dev, tokenizer, BS)

    wght_pattern = 'transformer.h.'
    wght_pattern_ln = ['ln_f.']
    if args.dev:
        rm_base = GPT2Model.from_pretrained('gpt2')

        train_params = tuple([wght_pattern+idx for idx in ['11']] + wght_pattern_ln)

        # freeze weights if parameter name starts with h.11
        for name, param in rm_base.named_parameters():
            if not name.startswith(train_params):
                param.requires_grad = False
    else:
        rm_base = GPT2Model.from_pretrained('geroldcsendes/sft-hh-rlhf')

    print('Trainable params:', f"{sum(p.numel() for p in rm_base.parameters() if p.requires_grad):,}")

    tech_args = dict(
        output_dir='ckp',
        do_train=True)

    if args.dev:
        per_device_effective_batch_size = 4
    else:
        per_device_effective_batch_size = 32
    gradient_accumulation_steps = per_device_effective_batch_size // BS

    lr = 6e-5
    num_epochs = 1

    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=False,
        **tech_args
    )

    if training_args.gradient_checkpointing:
        rm_base.gradient_checkpointing_enable()

    dt_now = datetime.now().strftime(format="%y-%m-%d-%H:%M:%S")
    run_name = f"{RUN_NAME}-{dt_now}"

    logdir = osp.join('logs', run_name)
    ckpdir = osp.join('ckp', run_name)
    writer = SummaryWriter(osp.join(logdir, 'tb'))

    if not os.path.exists(logdir):
        os.makedirs(logdir )
    if not os.path.exists(ckpdir):
        os.makedirs(ckpdir)
    shutil.copy(__file__, osp.join(logdir, 'rm.py'))

    print('writing logs to:', logdir)
    print('writing checkpoints to:', ckpdir)

    # set up opt process
    rm = RewardModel(base_model=rm_base)
    print('Reward model:\n', rm)
    rm = rm.to(device)

    optimizer = Adam(rm.parameters(), lr=lr)

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    num_save_steps = num_training_steps // 2

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps)

    pbar = tqdm(range(len(train_loader)))

    global_step = 0  # counter of ds batch level
    global_mb_step = 0  # counter of mb level (accumulated gradients)

    best_val_loss = 1e10

    for epoch in range(num_epochs):

        pbar = tqdm(range(len(train_loader)))
        running_loss = 0.0
        print('Running rm training')
        for batch in train_loader:

            loss = train_step(rm, batch, device)
            loss.backward()

            loss_item = loss.item()
            running_loss += loss_item

            pbar.set_description("step: %d, loss: %.3f" % (global_step, loss_item))
            pbar.update(1)

            if global_step % gradient_accumulation_steps == 0:
                # clip gradients
                t.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                running_loss /= gradient_accumulation_steps

                writer.add_scalar('loss_train', running_loss, global_mb_step)

                # write learning rate
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_mb_step)
                running_loss = 0.0

                global_mb_step += 1

            # scheduler.step()

            global_step += 1

            # run validation every n steps
            if global_step % num_save_steps == 0:  # num_save_steps
                running_val_loss = 0.0
                print('Running validation')
                for cnt, batch in enumerate(val_loader):
                    loss = valid_step(rm, batch, device)
                    loss_item = loss.item()

                    running_val_loss += loss_item

                running_val_loss /= len(val_loader)
                print('val loss:', running_val_loss)
                writer.add_scalar('loss_valid', running_val_loss, global_mb_step)

    pbar.close()
    rm.save_pretrained(logdir)
    tokenizer.save_pretrained(logdir)

    return

if __name__ == '__main__':
    main()