import datasets
from transformers import AutoTokenizer
# from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm, trange
import torch as t
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter
# from accelerate import Accelerator
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, Trainer, TrainingArguments
from transformers import GenerationConfig

from datetime import datetime
import os.path as osp
import shutil
from argparse import ArgumentParser

import os
import sys
from copy import deepcopy

from transformers import get_constant_schedule_with_warmup

from gpt2_ai.rlhf.rm import RewardModel
import torch as t
import torch as t


import logging

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_genconf(tokenizer, dev=False) -> dict:

    genconf = GenerationConfig(
        # generation parameters
        bos_token_id=tokenizer.bos_token_id,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=2.0,
        temperature=1.2,
        use_cache=True,

        # rlhf "parameters"
        num_return_sequences=1,
        max_new_tokens=30,  # TODO increase this later
        output_scores=True,
        return_dict_in_generate=True,  # output ModelOutput instead of tensors
    )

    genconf = genconf.to_dict()

    return genconf


def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='dev mode', default=False)

    return parser.parse_args()


def print_gpu_memory():
    print(f"GPU memory usage: {t.cuda.memory_allocated() / 1024**3:.2f} GB")


def get_loader(dev, tokenizer, bs=1) -> DataLoader:

    if dev:

        ds_list = [
            'What is the capital of France?',
            'I have a headache, please help me what to do.',
            'I feel like I want to hit someone. Is this okay?',
            'I struggle with math. What should I do?',
        ]

        ds_list = [f"\n\nHuman: {element}\n\nAssistant:" for element in ds_list]

        dataset = datasets.Dataset.from_dict({'prompt': ds_list})
    else:
        # TODO possibly append other datasets
        dataset = datasets.load_dataset('Dahoas/hh_human_eval')['train']

    def _encode_batch(batch):
        prompts = [element['prompt'] for element in batch]
        encoded = tokenizer(prompts,  padding=True, truncation=True,
                      return_tensors='pt', return_attention_mask=True)
        return encoded, prompts

    train_loader = t.utils.data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=_encode_batch,
        num_workers=4,
        pin_memory=True
    )

    return train_loader


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def compute_rewards(
        policy_logprobs: t.Tensor,
        sft_logprobs: t.Tensor,
        logprob_mask: t.Tensor,
        scores: t.Tensor,
        kl_coeff) -> Dict[str, t.Tensor]:

    # compute the KL divergence: idea taken from https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
    # reference: calculate approx_kl http://joschu.net/blog/kl-approx.html
    logratio = policy_logprobs - sft_logprobs  # [bs, gen_len]
    ratio = t.exp(logratio)  # [bs, gen_len]

    approx_kl = ((ratio -1) - logratio)
    # account for padding in generated tokens
    approx_kl = - approx_kl * logprob_mask
    approx_kl = kl_coeff * approx_kl



    logger.info(f"policy_logprobs shape: {policy_logprobs.shape}")
    logger.info(f"Approx KL shape: {approx_kl.shape}")

    reward = approx_kl.clone()
    logger.info(f"Reward shape: {reward.shape}")
    logger.info(f"Scores shape: {scores.shape}")
    # reward[:, last_nonpad_idx] += scores
    reward[:, -1] += scores

    return {
        'rewards': reward,
        'approx_kl': approx_kl,
        'rm_scores': scores}


def compute_advantages(
        rewards: t.Tensor, values: t.Tensor, gamma=0.99, lam=0.95,
        gae=False):

    if gae:
        # TODO implement GAE
        raise NotImplementedError
    else:
        # TODO implement optional discounting
        advantage = rewards - values

    return advantage


def batch_forward(
        policy, value_model,
        prompt_response,
        prompt_response_attn_mask,
        prompt_length,
        generated_tokens
        ) -> Dict:
    policy.eval()
    value_model.eval()

    logsoftmax = t.nn.LogSoftmax(dim=-1)

    policy_logits = policy(
        input_ids=prompt_response,  # [bs, gen_len] contains prompt and generated tokens
        attention_mask=prompt_response_attn_mask).logits

    # keep response logprobs only
    policy_logits = policy_logits[:, prompt_length:, :]  # [bs, gen_len, vocab_size]
    policy_logprobs = logsoftmax(policy_logits)  # [bs, gen_len, vocab_size]

    policy_logprobs = t.gather(  # [bs, gen_len]
        policy_logprobs, dim=-1,
        index=generated_tokens.unsqueeze(-1)).squeeze(-1)

    values = value_model.forward_value(
        input_ids=prompt_response,
        attention_mask=prompt_response_attn_mask)

    print(f"{values.shape=}")

    # keep response values only
    values = values[:, prompt_length:, :].squeeze(-1)  # [bs, gen_len]

    return {'policy_logprobs': policy_logprobs, 'values': values}



class RMMock(t.nn.Module):
    """Dummy RewardModel that returns minmax scaled length (excluding the padding tokens)
    of the generated text"""
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, **kwargs):
        # get reward for 'ale' token
        #reward = t.sum(input_ids == 1000, dim=1, keepdim=True).float()
        reward = (input_ids == 1000).float()
        reward += t.rand_like(reward) * 0.1

        return reward

    def forward_value(self, input_ids, **kwargs):
        # get reward for 'ale' token
        #reward = t.sum(input_ids == 1000, dim=1, keepdim=True).float()
        reward = (input_ids == 1000).float()
        reward += t.rand_like(reward) * 0.1

        return reward.unsqueeze(-1)

    def gradient_checkpointing_enable(self):
        pass


@dataclass
class BatchContainer:
    query_response: t.Tensor
    query_response_attn_mask: t.Tensor
    policy_logprobs: t.Tensor
    values: t.Tensor
    advantages: t.Tensor
    approx_kl: t.Tensor
    returns: t.Tensor
    reward_scores: t.Tensor


@dataclass
class PPOConfig:
    ppo_batch_size: int
    ppo_epochs: int = 1
    learning_rate: float = 1e-5
    gamma: float = 0.99
    lam: float = 0.95
    gae: bool = False
    kl_coeff: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    max_grad_norm: float = 0.5
    vf_coef: float = 0.1
    learning_rate: float = 1e-5


@dataclass
class TrainStats:
    policy_loss: float
    value_loss: float
    returns: Optional[float] = None
    values: Optional[float] = None
    advantages: Optional[float] = None
    approx_kl: Optional[float] = None
    reward_scores: Optional[float] = None


def train_minibatch(
        policy,
        optimizer,
        old_logprobs: t.Tensor,
        old_values: t.Tensor,
        logprobs: t.Tensor,
        logprobs_mask: t.Tensor,
        values: t.Tensor,
        advantages: t.Tensor,
        returns: t.Tensor,
        conf: PPOConfig):

    policy.train()

    #region pg loss
    ratio = t.exp(logprobs - old_logprobs)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * t.clamp(
        ratio, 1 - conf.cliprange, 1 + conf.cliprange)

    pg_loss = masked_mean(
        t.max(pg_losses, pg_losses2),
        logprobs_mask, axis=1)

    #endregion pg loss

    #region value loss
    vpredclipped = old_values + t.clamp(
        values - old_values,
        -conf.cliprange_value,
        conf.cliprange_value)

    vf_losses1 = t.square(values - returns)
    vf_losses2 = t.square(vpredclipped - returns)
    vf_loss = 0.5 * masked_mean(t.max(vf_losses1, vf_losses2), logprobs_mask)

    # clip vf loss
    vf_loss = t.clamp(vf_loss, 0, 2)
    #endregion value loss

    loss = pg_loss + vf_loss * conf.vf_coef
    loss = loss.mean()
    loss.backward()
    t.nn.utils.clip_grad_norm_(policy.parameters(), conf.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return TrainStats(
        policy_loss=pg_loss.mean().item(),
        value_loss=vf_loss.mean().item())


if __name__ == '__main__':

    from huggingface_hub import hf_hub_download
    rm_linpath = hf_hub_download(repo_id="geroldcsendes/rm-hh-rlhf", filename="pytorch_model_lin.bin")
    # print(rm_linpath)
    # import sys
    # sys.exit(0)

    #region tboard
    RUN_NAME = 'RLHF/RLHF-dev'

    dt_now = datetime.now().strftime(format="%y-%m-%d-%H:%M:%S")
    run_name = f"{RUN_NAME}-{dt_now}"

    logdir = osp.join('logs', run_name)
    ckpdir = osp.join('ckp', run_name)
    writer = SummaryWriter(osp.join(logdir, 'tb'))
    #endregion tboard

    logger = setup_logger('my.log')

    args = parse_args()

    # higher BS can be used because this will only be used for inference (
    # create experience)
    if args.dev:
        BS = 2
    else:
        BS = 4

    ppo_conf = PPOConfig(ppo_batch_size=BS / 2)

    RUN_NAME = 'sft-hh-rlhf'

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    # get max vram

    # left padding needed for generation
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_loader(
        dev=args.dev, tokenizer=tokenizer, bs=BS)

    # print(next(iter(train_loader)))

    # get the sft model
    sft = GPT2LMHeadModel.from_pretrained('geroldcsendes/sft-hh-rlhf')

    # init the policy from the sft
    policy = GPT2LMHeadModel.from_pretrained('geroldcsendes/sft-hh-rlhf')
    policy.to(device)

    # get the rm model
    if args.dev:
        rm = RMMock()
        # rm = RewardModel.from_pretrained('geroldcsendes/rm-hh-rlhf', device=device,
        #                                  pretrained_lin_path=rm_linpath)
    else:
        rm = RewardModel.from_pretrained('geroldcsendes/rm-hh-rlhf', device=device,
                                         pretrained_lin_path=rm_linpath)

    # initialize the value function from the reward model
    if args.dev:
        value = RMMock()
        # value = RewardModel.from_pretrained('geroldcsendes/rm-hh-rlhf', device=device,
        #                                     pretrained_lin_path=rm_linpath)
    else:
        value = RewardModel.from_pretrained('geroldcsendes/rm-hh-rlhf', device=device,
                                            pretrained_lin_path=rm_linpath)
    value.to(device)

    # freeze the reward model
    for param in rm.parameters():
        param.requires_grad = False
    rm.to(device)

    # freeze the sft
    for param in sft.parameters():
        param.requires_grad = False
    sft.to(device)

    if device == 'cuda':
        print_gpu_memory()

    # Set up optimizer for policy and value models
    optimizer = t.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=ppo_conf.learning_rate)

    # enable gradient checkpointing for policy and value models
    policy.gradient_checkpointing_enable()
    value.gradient_checkpointing_enable()

    genconf = get_genconf(tokenizer, dev=False)

    num_updates = 100
    num_steps = 100

    # init storage tensors
    observation_space = tokenizer.vocab_size
    obs = t.zeros(num_steps, observation_space)

    logsoftmax = t.nn.LogSoftmax(dim=-1)

    global_step = 0
    for epoch in range(num_updates):
        for prompt, prompt_str in train_loader:
            # prompt is a dict with keys: ['input_ids', 'attention_mask']
            for key, v in prompt.items():
                logger.info(f'{key}: {v.shape}')

            for _ in prompt_str:
                logger.info(f'Prompt: {_}')

            # put prompt on device
            for key, v in prompt.items():
                prompt[key] = v.to(device)

            #region policy
            with t.no_grad():
                policy.eval()
                policy_output = policy.generate(
                    **prompt,
                    **genconf)

            logger.info(f"Generated sequences shape: {policy_output.sequences.shape}")
            logger.info(f"Generated scores shape: {policy_output.scores[0].shape}")

            prompt_length = prompt['input_ids'].shape[1]
            generated_tokens = policy_output.sequences[:, prompt_length:]
            logger.info(f"{generated_tokens.shape=}")

            # create attention mask for policy forward where the prompt pad is masked
            to_pad = policy_output.sequences.shape[1] - prompt['attention_mask'].shape[1]
            prompt_response_attn_mask = t.cat(
                [prompt['attention_mask'], t.ones(BS, to_pad, device=device)], dim=1)

            # get logprobs from policy
            with t.no_grad():
                policy.eval()
                policy_logits = policy(
                    input_ids=policy_output.sequences,  # [bs, gen_len] contains prompt and generated tokens
                    attention_mask=prompt_response_attn_mask).logits

            # keep response logprobs only
            policy_logits = policy_logits[:, prompt_length:, :]  # [bs, gen_len, vocab_size]
            policy_logprobs = logsoftmax(policy_logits)  # [bs, gen_len, vocab_size]

            policy_logprobs = t.gather(  # [bs, gen_len]
                policy_logprobs, dim=-1,
                index=generated_tokens.unsqueeze(-1)).squeeze(-1)

            # get mask for padding tokens - needed for the KL divergence loss
            logprob_mask: t.Tensor = generated_tokens != tokenizer.pad_token_id
            logprob_mask = logprob_mask.long()

            logger.info(f"{policy_logprobs.shape=}")
            logger.info(f"{policy_logprobs=}")

            #endregion policy

            #region sft
            with t.no_grad():
                sft.eval()
                sft_logits = sft(
                    input_ids=policy_output.sequences,  # [bs, gen_len] contains prompt and generated tokens
                    attention_mask=prompt_response_attn_mask).logits

            # keep response logprobs only
            sft_logits = sft_logits[:, prompt_length:, :]  # [bs, gen_len, vocab_size]
            sft_logprobs = logsoftmax(sft_logits)  # [bs, gen_len, vocab_size]

            sft_logprobs = t.gather(
                sft_logprobs, dim=-1,
                index=generated_tokens.unsqueeze(-1)).squeeze(-1)  # [bs, gen_len]

            logger.info(f"SFT logprobs: {sft_logprobs}")

            assert sft_logprobs.shape == policy_logprobs.shape, \
                f"Shapes of logprobs and sft_logprobs do not match: {sft_logprobs.shape} \
                vs {policy_logprobs.shape}"
            #endregion sft

            # region helper print
            # batch decode the generated sequences
            decoded = tokenizer.batch_decode(policy_output.sequences, skip_special_tokens=True)
            for _ in decoded:
                logger.info(f"Generated decoded sequence: {_}")

            logger.info(f"Generated token ids: {policy_output.sequences}")
            # endregion helper print

            #region rm
            # last nonpad idx should be the same as the length of the generated tokens
            # because the generated tokens are left padded which is handled
            # with the attention mask in RM
            last_nonpad_idx = (prompt_response_attn_mask.shape[1] - 1) \
                * t.ones(BS, dtype=t.long, device=device)

            logger.info(f"{last_nonpad_idx=}")
            logger.info(f"Last nonpad index shape: {last_nonpad_idx.shape}")

            logger.info(f"{prompt_response_attn_mask=}")
            logger.info(f"{prompt_response_attn_mask.shape=}")
            # get the values for the generated sequences
            with t.no_grad():
                rm.eval()

                scores: t.Tensor = rm(
                    input_ids=policy_output.sequences,  # contains prompt and generated tokens
                    attention_mask=prompt_response_attn_mask,  # mask prompt left padding
                    last_nonpad_idx=last_nonpad_idx)

            # only keep score for the full generated sequence
            scores = scores[:, -1] # [bs]
            logger.info(f"{scores=}")
            logger.info(f"{scores.shape=}")

            #endregion rm

            #region value
            with t.no_grad():
                value.eval()
                values = value.forward_value(
                    input_ids=policy_output.sequences,
                    attention_mask=prompt_response_attn_mask)

            # keep response values only
            logger.info(f"PRE-{values.shape=}")
            values = values[:, prompt_length:].squeeze(-1)  # [bs, gen_len, vocab_size]
            logger.info(f"{values.shape=}")

            #endregion value

            #region compute rewards
            out = compute_rewards(
                policy_logprobs=policy_logprobs,
                sft_logprobs=sft_logprobs,
                logprob_mask=logprob_mask,
                scores=scores,
                kl_coeff=0.02)

            logger.info(f"Rewards: {out['rewards']}")
            logger.info(f"Rewards shape: {out['rewards'].shape}")
            logger.info(f"Approx KL: {out['approx_kl']}")
            logger.info(f"Approx KL shape: {out['approx_kl'].shape}")
            #endregion compute rewards

            #region compute advantages
            advantage = compute_advantages(
                rewards=out['rewards'],
                values=values)

            #endregion compute advantages

            batch_container = BatchContainer(
                query_response=policy_output.sequences,
                query_response_attn_mask=prompt_response_attn_mask,
                policy_logprobs=policy_logprobs,
                values=values,
                advantages=advantage,
                approx_kl=out['approx_kl'],
                returns=out['rewards'],
                reward_scores=out['rm_scores']
                )

            #region ppo update
            # TODO implement PPO update
            for _ in range(ppo_conf.ppo_epochs):
                b_inds = np.random.permutation(BS)
                b_inds = np.array_split(b_inds, BS / ppo_conf.ppo_batch_size)

                for b_ind in b_inds:
                    minibatch_container = BatchContainer(
                        query_response=batch_container.query_response[b_ind],
                        query_response_attn_mask=batch_container.query_response_attn_mask[b_ind],
                        policy_logprobs=batch_container.policy_logprobs[b_ind],
                        values=batch_container.values[b_ind],
                        advantages=batch_container.advantages[b_ind],
                        approx_kl=batch_container.approx_kl[b_ind],
                        returns=batch_container.returns[b_ind],
                        reward_scores=batch_container.reward_scores[b_ind])

                    logger.info(f"MB query-resonse shape: {minibatch_container.query_response.shape}")
                    mb_out = batch_forward(
                        policy=policy,
                        value_model=value,
                        prompt_response=minibatch_container.query_response,
                        prompt_response_attn_mask=minibatch_container.query_response_attn_mask,
                        prompt_length=prompt_length,
                        generated_tokens=generated_tokens[b_ind])

                    logger.info(f"MB policy logprobs shape: {mb_out['policy_logprobs'].shape}")
                    logger.info(f"MB values shape: {mb_out['values'].shape}")
                    logger.info(f"MB values: {mb_out['values']}")

                    # sys.exit(0)

                    stats = train_minibatch(
                        policy=policy,
                        optimizer=optimizer,
                        old_logprobs=minibatch_container.policy_logprobs,
                        old_values=minibatch_container.values,
                        logprobs=mb_out['policy_logprobs'],
                        logprobs_mask=logprob_mask[b_ind],
                        values=mb_out['values'],
                        advantages=minibatch_container.advantages,
                        returns=minibatch_container.returns,
                        conf=ppo_conf)

                    stats.returns = minibatch_container.returns.mean().item()
                    stats.values = minibatch_container.values.mean().item()
                    stats.advantages = minibatch_container.advantages.mean().item()
                    stats.approx_kl = minibatch_container.approx_kl.mean().item()
                    stats.reward_scores = minibatch_container.reward_scores.mean().item()

                    logger.info(f"Stats: {stats}")

                    for k, v in stats.__dict__.items():
                        writer.add_scalar(k, v, global_step=global_step)

                    global_step += 1

