from datetime import datetime
import os.path as osp
import os

from tqdm import tqdm
import torch as t
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter

from gpt2_ai.config import GPT2Config
from gpt2_ai.model import GPT2
from gpt2_ai.util.names import get_random_name


class Trainer:
    def __init__(self, config: GPT2Config, model: GPT2,
                 train_loader: DataLoader, tokenizer: GPT2Tokenizer,
                 valid_loader=None):
        
        self.config = config
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
        self.writer = SummaryWriter(osp.join(self.config.log_path, run_name))

        self.ckpt_path = osp.join(self.config.ckpt_path, dt_now)
        if not osp.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

    def train_step(self, batch: Tensor) -> Tensor:
        self.model.train()

        logits = self.model(batch)[:,:-1,:]
        logits = logits.reshape(-1, self.tokenizer.vocab_size)  # [batch_size*seq_len, vocab_size]
        target = batch[:,1:].reshape(-1)  # [batch_size*seq_len]
        
        loss = self.criterion(logits, target)

        return loss
    
    @t.inference_mode()
    def valid_step(self, batch: Tensor) -> Tensor:
        self.model.eval()
        return
    
    def train(self):
        
        model = self.model
        opt = self.optimizer(model.parameters(), lr=self.config.lr)

        step = 0
        for epoch in tqdm(range(self.config.n_epochs)):
            for batch in self.train_loader:
                
                opt.zero_grad()
                batch = batch['input_ids'].to(self.config.device)

                loss = self.train_step(batch)
                loss.backward()
                opt.step()

                if step % self.config.log_interval == 0:
                    self.writer.add_scalar('loss', loss.item(), step)

                if step % self.config.ckpt_interval == 0:
                    t.save(self.model.state_dict(),
                           f"{self.ckpt_path}/step-{step}.pt")

                step += 1

                print(f"Step: {step} | Loss: {loss.item():.3f}")

        return
