# GPT2-ALL-IN

This repository is intended to be a one-stop-shop for GPT2-related to train and finetune, and benchmark models. It also provides utilities to orchestrate the training and finetuning process on remote GPUs on LambdaLabs.

This project may be helpful for those who are relatively new to GPT2-style models and would like get a deep dive. I learnt a bunch of things while working on this project and I hope this repository will help you learn a few things too.

This repository draws inspiration from:
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- [ARENA](https://www.arena.education/) (Alignment Research Engineer Accelerator): [github](https://github.com/callummcdougall/ARENA_2.0)

## Things you can find here and WIP

- Semi-clean GPT2 implementation
- Configs to control training and finetuning and to persist them
- Demo notebooks
- Utilities to use LambdaLabs infra
    - set up environment
    - logging
    - persisting checkpoints
- Switch to use development mode for faster debugging
