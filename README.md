# GPT2-ALL-IN

This repository is intended to be a one-stop-shop for GPT2-related to train and finetune, and benchmark models. It also provides utilities to orchestrate the training and finetuning process on remote GPUs on LambdaLabs.

This project may be helpful for those who are relatively new to GPT2-style models and would like get a deep dive. I learnt a bunch of things while working on this project and I hope this repository will help you learn a few things too.

This repository draws inspiration from:
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- [ARENA](https://www.arena.education/) (Alignment Research Engineer Accelerator): [github](https://github.com/callummcdougall/ARENA_2.0)

## Things you can find here and WIP

- Semi-clean/optimized GPT2 implementation
- Configs to control training and finetuning and to persist them
- Demo notebooks
- Utilities to use LambdaLabs infra
    - set up environment
    - logging
    - persisting checkpoints
- Switch to use development mode for faster debugging

## Environment setup
You have 3 options to set up your environment:
1. Use pyenv and poetry to create a Python venv
1. Use the `environment.yml` file to create a conda environment
1. Use the `Dockerfile` to create a docker image

What is cool is that you will actually run the same code both on your local machine and on LambdaLabs. Thus, you should not have to worry about environment issues if all goes well.

I personally like to use pyenv and poetry to manage my environments. I find it to be the most flexible and the most lightweight and very handy for local debugging. However, it takes some time to learn it and set it up. If you are not familiar with it, I recommend you use the conda or docker option.

### Using pyenv and poetry
Run the following make targets to set up your environment:
- `make setup-pyenv`
- `make install-python`
- `make install-poetry`
- `make install-rust` # needed to install transformer tokenizers

### Using conda
TODO

## How-to-train
TODO

### On local machine w/CPU
TODO

### On lambdalabs w/GPU
TODO