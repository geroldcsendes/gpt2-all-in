from datetime import datetime
from functools import wraps
import os.path as osp
import os

from tqdm import tqdm
import torch as t
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter

from gpt2_ai.train.config import GPT2Config, TrainerConfig
from gpt2_ai.train.model import GPT2
from gpt2_ai.util.names import get_random_name


def use_bfloat16(func):
    """
    Decorator for using bfloat16 precision.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.trainer_config.bfloat16:
            with t.cuda.amp.autocast(dtype=t.bfloat16):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return wrapper


class Trainer:
    def __init__(self, config: GPT2Config, model: GPT2,
                 trainer_config: TrainerConfig,
                 train_loader: DataLoader, tokenizer: GPT2Tokenizer,
                 valid_loader=None):

        self.config = config
        self.trainer_config = trainer_config
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = getattr(t.optim, config.optimizer)
        self.criterion = t.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        self.writer = None
        self.ckpt_path = None

        self._init_log()

    def _init_log(self):
        dt_now = datetime.now().strftime(format="%y%m%d%H")
        run_name = f"{get_random_name()}-{dt_now}"
        self.writer = SummaryWriter(osp.join(self.config.log_path, run_name, 'tb'))

        self.ckpt_path = osp.join(self.config.ckpt_path, run_name, 'ckpt')
        if not osp.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.run_name = run_name

    @use_bfloat16
    def train_step(self, batch: Tensor) -> Tensor:
        self.model.train()

        logits = self.model(batch)[:,:-1,:]
        logits = logits.reshape(-1, self.tokenizer.vocab_size)  # [batch_size*seq_len, vocab_size]
        target = batch[:,1:].reshape(-1)  # [batch_size*seq_len]

        loss = self.criterion(logits, target)

        return loss

    @t.inference_mode()
    @use_bfloat16
    def valid_step(self, batch: Tensor) -> Tensor:
        self.model.eval()

        logits = self.model(batch)[:,:-1,:]
        logits = logits.reshape(-1, self.tokenizer.vocab_size)  # [batch_size*seq_len, vocab_size]
        target = batch[:,1:].reshape(-1)  # [batch_size*seq_len]

        loss = self.criterion(logits, target)

        return loss

    def log_ckpt(self, step: int, loss: Tensor):

        if step % self.config.log_interval == 0:
            self.writer.add_scalar('loss_train', loss, step)

        if step % self.config.ckpt_interval == 0:
            t.save(self.model.state_dict(),
                f"{self.ckpt_path}/step-{step}.pt")

        if step % self.config.valid_interval == 0  and self.valid_loader is not None:
            for valid_batch in self.valid_loader:
                valid_batch = valid_batch['input_ids'].to(self.config.device)
                valid_loss = self.valid_step(valid_batch)
            self.writer.add_scalar('loss_valid', valid_loss.item(), step)

        step += 1

        print(f"Step: {step} | Loss: {loss.item():.3f}")

        return

    def train(self):

        train_config = self.trainer_config
        model = self.model
        opt = self.optimizer(model.parameters(), lr=self.config.lr)

        step = 0
        for _ in tqdm(range(self.config.n_epochs)):

            for batch in self.train_loader:

                batch = batch['input_ids'].to(self.config.device)

                loss = self.train_step(batch)
                loss.backward()

                self.log_ckpt(step, loss)

                # in case of gradient accumulation, check if enough steps passed
                if train_config.gradient_accumulation_steps:
                    if step % train_config.gradient_accumulation_steps == 0:
                        opt.step()
                    else:
                        continue  # skip the rest of the loop

                opt.step()
                opt.zero_grad()

        return

