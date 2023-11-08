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
    # take half of the dataset
    dataset = dataset.shuffle(seed=1).select(range(len(dataset)//2))
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
        input_ids_chosen = tokenizer(chosen,  padding=True, truncation=True,
                      return_tensors='pt', return_attention_mask=False)

        return input_ids_chosen

    train_loader = t.utils.data.DataLoader(
        train, batch_size=bs, shuffle=True, collate_fn=_encode_batch,
        num_workers=4, pin_memory=True)
    val_loader = t.utils.data.DataLoader(
        val, batch_size=2*bs, shuffle=True, collate_fn=_encode_batch,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader


def train_step(model, batch, tokenizer, criterion):
    model.train()

    logits = model(**batch).logits[:,:-1,:]
    logits = logits.reshape(-1, tokenizer.vocab_size)  # [batch_size*seq_len, vocab_size]
    target = batch['input_ids'][:,1:].reshape(-1)  # [batch_size*seq_len]

    loss = criterion(logits, target)

    return loss

@t.inference_mode()
def valid_step(model, batch, tokenizer, criterion):
    model.eval()

    logits = model(**batch).logits[:,:-1,:]
    logits = logits.reshape(-1, tokenizer.vocab_size)  # [batch_size*seq_len, vocab_size]
    target = batch['input_ids'][:,1:].reshape(-1)  # [batch_size*seq_len]

    loss = criterion(logits, target)

    return loss


def main():

    args = parse_args()

    if args.dev:
        BS = 2
    else:
        BS = 4

    RUN_NAME = 'SFT/SFT-dev'

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    # get max vram

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = get_loader(args.dev, tokenizer, BS)

    wght_pattern = 'transformer.h.'
    wght_pattern_ln = ['ln_f.']
    if args.dev:
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        train_params = tuple([wght_pattern+idx for idx in ['11']] + wght_pattern_ln)

        # freeze weights if parameter name starts with h.11
        for name, param in model.named_parameters():
            if not name.startswith(train_params):
                param.requires_grad = False
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

        train_params = tuple([wght_pattern+str(idx) for idx in range(20, 25)] + wght_pattern_ln)

        for name, param in model.named_parameters():
            if name.startswith(train_params):
                param.requires_grad = False

    print('Trainable params:', f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    tech_args = dict(
        output_dir='ckp',
        do_train=True)

    if args.dev:
        per_device_effective_batch_size = 4
    else:
        per_device_effective_batch_size = 64
    gradient_accumulation_steps = per_device_effective_batch_size // BS

    lr = 6e-5
    num_epochs = 1

    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        fp16=False,
        **tech_args
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = Adam(model.parameters(), lr=lr)

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    num_save_steps = num_training_steps // 4

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps)

    # sample = next(iter(train_loader)).to(device)
    model = model.to(device)
    # out = model(**sample)

    dt_now = datetime.now().strftime(format="%y-%m-%d-%H:%M:%S")
    run_name = f"{RUN_NAME}-{dt_now}"

    logdir = osp.join('logs', run_name)
    ckpdir = osp.join('ckp', run_name)
    writer = SummaryWriter(osp.join(logdir, 'tb'))

    if not os.path.exists(logdir):
        os.makedirs(logdir )
    if not os.path.exists(ckpdir):
        os.makedirs(ckpdir)
    shutil.copy(__file__, osp.join(ckpdir, 'sft.py'))

    print('writing logs to:', logdir)
    print('writing checkpoints to:', ckpdir)

    criterion = t.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    pbar = tqdm(range(len(train_loader)))

    global_step = 0  # counter of ds batch level
    global_mb_step = 0  # counter of mb level (accumulated gradients)

    best_val_loss = 1e10

    for epoch in range(num_epochs):

        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            loss = train_step(model, batch, tokenizer, criterion)
            loss.backward()

            loss_item = loss.item()
            running_loss += loss_item

            pbar.set_description("step: %d, loss: %.3f" % (global_step, loss_item))
            pbar.update(1)

            if global_step % gradient_accumulation_steps == 0:
                global_mb_step += 1

                optimizer.step()
                optimizer.zero_grad()

                running_loss /= gradient_accumulation_steps

                writer.add_scalar('loss_train', running_loss, global_mb_step)
                # write learning rate
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_mb_step)
                running_loss = 0.0

            scheduler.step()

            global_step += 1

            # run validation every n steps
            if global_step % num_save_steps == 0:  # num_save_steps
                running_val_loss = 0.0
                print('running validation')
                for cnt, batch in enumerate(val_loader):
                    batch = batch.to(device)

                    loss = valid_step(model, batch, tokenizer, criterion)
                    loss_item = loss.item()

                    running_val_loss += loss_item

                running_val_loss /= len(val_loader)
                print('val loss:', running_val_loss)
                writer.add_scalar('loss_valid', running_val_loss, global_mb_step)

                if running_val_loss < best_val_loss:
                    best_val_loss = running_val_loss
                    print('saving checkpoint')
                    t.save(
                        {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch
                        },
                        f"{ckpdir}/step-{global_mb_step}.pt")

    return

if __name__ == '__main__':
    main()