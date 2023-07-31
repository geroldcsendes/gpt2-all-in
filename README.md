# GPT2-ALL-IN

This repository is intended to be a one-stop-shop for GPT2-related to train and finetune, and benchmark models. It also provides utilities to orchestrate the training and finetuning process on remote GPUs on LambdaLabs.

This project may be helpful for those who are relatively new to GPT2-style models and would like get a deep dive. I learnt a bunch of things while working on this project and I hope this repository will help you learn a few things too.

This repository draws inspiration from:
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- [ARENA](https://www.arena.education/) (Alignment Research Engineer Accelerator): [github](https://github.com/callummcdougall/ARENA_2.0)

## Things you can find here

- Semi-clean/optimized GPT2 implementation
- Configs to control training and finetuning and to persist them
- A demo notebook
- Utilities to use LambdaLabs infra
    - set up environment
    - logging
    - persisting checkpoints
- Switch to use development mode for faster debugging

## Environment setup
You have 3 options to set up your environment:
1. Use pyenv and poetry to create a Python venv
1. Use the `environment.yml` file to create a conda environment
1. Use the `Dockerfile` to create a docker image (TODO)

What is cool is that you will actually run the same code both on your local machine and on LambdaLabs. Thus, you should not have to worry about environment issues if all goes well.

I personally like to use pyenv and poetry to manage my environments. I find it to be the most flexible and the most lightweight and very handy for local debugging. However, it takes some time to learn it and set it up. If you are not familiar with 
it, I recommend you use the conda or docker option.

You probably don't have to rul all the below steps for your local environment setup, assuming that you already have Pyenv + poetry / conda / docker installed. In that case, you can get by only running:

Pyenv + poetry:
```bash
poetry install
```

Conda:
```bash
make create-conda-env
conda activate gpt2-ai
```

Docker: TODO

For setting up your environment remotely, follow the instructions below.

Steps:
1. Launch a LambdaLabs GPU instance
2. SSH into the instance
3. Clone and cd into this repository via: `git clone https://github.com/geroldcsendes/gpt2-all-in.git && cd gpt2-all-in`

### Using pyenv and poetry
4. Run `setup-pyenv-poetry` to set up the environment and install the dependencies
5. Run `poetry shell` to activate the environment

### Using conda
4. Run `create-conda-env`
5. Run `conda activate gpt2-ai`

### Using docker
TODO

## How-to-train
Training and model configurations are parsed from the configs/ directory. You can find 2 reference configs there: `cpu.json`,  `gpu.json`.

NOTE that the device will be picked automatically from `torch.cuda.is_available()` and not from the configuration file. The cpu and gpu configs are there for reference only. The sample cpu config defines a small model that can be trained on a CPU. The gpu config defines a larger model that can be trained on a GPU.

The configs should have 3 attributes: dataset, model and trainer as shown below:

```json
{
    "dataset": "full",
    "model": 
        {
            "n_head": 8,
            "n_layer": 4,
            "n_ctx": 24,
            "d_model": 64
        },
    "trainer":
        {
            "n_epochs": 1
        }
}
```

The dataset attribute is used to select the dataset to train on. For now, only two values are allowed: dev and full. The dev dataset is a small dataset that is used for debugging purposes and is mapped to [stas/openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k). The full dataset is the full dataset that is used for a 'proper' training and is mapped to [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext).

The model attribute is used to construct the model configuration and is parsed to initialize the `GPT2Config` at _config.py_. 

The trainer attribute is used to configure the training process and is parsed to initialize the `TrainerConfig` at __config.py__. 

Default values in these two configuration classes will get overwritten by the values in the config file. You may experiment what configurations works best on which dataset and GPU.

Having set the configs, you can start training by:
```bash
python train.py --config {your-config-here}.json
```

### Logging
A unique log directory will be created for each run under logs/. The following naming convention is used: <run-name>-<%y%m%d%H> e.g. jolly_ptolemy-23073110. The run name is created automatically and draws inspiration from Docker's fun container names. 

The log directory consists of 1) a checkpoint directory and 2) a tensorboard directory and 3) a the training config file (for reproducibility). Here is an example log of a training run:

```
├── jolly_ptolemy-23073110
│   ├── ckpt
│   ├── config.json
│   └── tb
```