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