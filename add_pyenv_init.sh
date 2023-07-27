#!/bin/bash

rc_files=("$HOME/.bashrc" "$HOME/.zshrc")

for rc_file in ${rc_files[@]}; do
  if [[ -f $rc_file ]]; then
    FILE_FOUND=1
    if [[ -z $(grep 'eval "$(pyenv init -)"' $rc_file) ]]; then
      echo "Adding pyenv initialization to $rc_file"
      cat ./enable-pyenv.sh >> $rc_file
    else
      echo "Pyenv initialization is found in $rc_file"
    fi
  fi
done

if [[ -z $FILE_FOUND ]]; then
  echo "Please add the following lines to your .bashrc file:"
  echo
  cat ./enable-pyenv.sh
fi
