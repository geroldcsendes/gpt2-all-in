SHELL := /bin/bash
poetry = poetry

CONDA_HOME = $(HOME)/miniconda3
CONDA_BIN_DIR = $(CONDA_HOME)/bin
CONDA = $(CONDA_BIN_DIR)/conda

# Env: Pyenv + Poetry
install-apt-dependencies:
	sudo apt-get update && sudo apt-get install -y build-essential cmake pkg-config gcc zlib1g-dev libbz2-dev libssl-dev libreadline-dev libsqlite3-dev libfreetype6-dev libblas-dev liblapack-dev gfortran wget curl libncurses5-dev xz-utils libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev jq zip unzip git

setup-pyenv-poetry: install-apt-dependencies 
	@curl https://pyenv.run | bash;\
	/bin/bash ./add-pyenv-init.sh;
	@source ./enable-pyenv.sh && \
	pyenv install 3.10.7 && \
	pyenv global 3.10.7 && \
	echo "Python 3.10.7 installation completed." && \
	echo "Installing Poetry 1.2.2..." && \
	pip install --upgrade pip && \
	pip install poetry==1.2.2 && \
	poetry --version && \
	echo "Poetry 1.2.2 installation completed." && \
	echo "Installing Rust for Huggingface tokenizer .." && \
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
	source $(HOME)/.cargo/env && \
	poetry install

# Env: Conda
install-conda:
	@echo "Installing Conda..."
	@mkdir -p ~/miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	@bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	@rm -rf ~/miniconda3/miniconda.sh
	@~/miniconda3/bin/conda init bash
	@source ~/.bashrc
	@echo "Conda installation completed."

create-conda-env:
	@echo "Creating conda env..."
	$(CONDA) env create -f env.yml
	@echo "Conda env creation completed."

setup-conda: install-conda create-conda-env

# Other util stuff
clone-cd:
	git clone https://github.com/geroldcsendes/gpt2-all-in.git && cd gpt2-all-in

register-jupyter:
	@echo "--- [Registering venv to jupyter using ipykernel..] ---"
	$(poetry) run python -m ipykernel  install --user --name=gpt2

# config git so you can push to the remote
git-config:
	git config user.email gerold.csendes@gmail.com
	git config user.name geroldcsendes

# start remote tensorboard e.g. ip=127.000.000.00
tensorboard-remote:
	ssh -N -f -L localhost:16006:localhost:6006 ubuntu@$(ip)

check-gpu:
	watch nvidia-smi

clean-env:
	rm -rf .venv
	rm -rf *.egg-info
	rm -rf .pytest_cache

clean-ckpt-log:
	rm -rf ckpt
	rm -rf log
