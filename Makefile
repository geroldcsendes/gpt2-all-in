poetry = poetry

conda-update:
	conda env update -f env.yml

install:
	poetry install

register-jupyter:
	@echo "--- [Registering venv to jupyter using ipykernel..] ---"
	$(poetry) run python -m ipykernel  install --user --name=gpt2

install-pyenv:
	@if [ ! -d ~/.pyenv ]; then curl https://pyenv.run | bash; fi;
	@/bin/bash ./add-pyenv-init.sh;
	@source ./enable-pyenv.sh && \
	pyenv uninstall -f 3.10.7 && pyenv install -f 3.10.7 && \
	pyenv uninstall -f 3.8.12 && pyenv install -f 3.8.12 && \
	rm -f ${HOME}/.local/bin/poetry && \
	pyenv shell 3.8.12 && \
	pip install -U pip poetry==1.2.2 pre-commit && \
	pyenv shell --unset && \
	pyenv global 3.10.7 && \
	pip install -U pip poetry==1.2.2 pre-commit && \
	poetry config virtualenvs.in-project true


SHELL := /bin/bash
# Makefile to install and set up pyenv, Python 3.10.7, and Poetry 1.2.2 on the local machine

install-apt-dependencies:
	sudo apt-get update && sudo apt-get install -y build-essential cmake pkg-config gcc zlib1g-dev libbz2-dev libssl-dev libreadline-dev libsqlite3-dev libfreetype6-dev libblas-dev liblapack-dev gfortran wget curl libncurses5-dev xz-utils libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev jq zip unzip git

PYENV_INSTALL_COMMANDS = \
	curl https://pyenv.run | bash; \
	echo 'export PATH="$$HOME/.pyenv/bin:$$PATH"' >> ~/.bashrc; \
	echo 'eval "$$(pyenv init -)"' >> ~/.bashrc; \
	echo 'eval "$$(pyenv virtualenv-init -)"' >> ~/.bashrc; \
	source ~/.bashrc;

setup-pyenv:
	@echo "Installing and setting up pyenv..."
	$(PYENV_INSTALL_COMMANDS)
	@echo "Pyenv installation and setup completed."

install-python:
	@echo "Installing Python 3.10.7..."
	pyenv install 3.10.7
	pyenv global 3.10.7
	@echo "Python 3.10.7 installation completed."

install-poetry:
	@echo "Installing Poetry 1.2.2..."
	pip install --upgrade pip
	pip install poetry==1.2.2
	poetry --version
	@echo "Poetry 1.2.2 installation completed."

# this is needed for huggingface tokenizer
install-rust:
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

poetry-install:
	poetry install

setup-all: install-apt-dependencies setup-pyenv install-python install-poetry install-rust
	@echo "Setup completed."