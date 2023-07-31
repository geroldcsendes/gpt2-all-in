# Set up pynev
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
# Load pyenv-virtualenv automatically by uncommenting:
# eval "$(pyenv virtualenv-init -)"
