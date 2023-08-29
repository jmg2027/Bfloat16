# This is reference!
#!/bin/bash

source ~/.bashrc

SCRIPT_DIR=$(realpath $(dirname $BASH_SOURCE))
TOP_DIR=$(realpath $SCRIPT_DIR/..)
PS_NAME=$(basename $TOP_DIR)
CUSTOM_PS1="($PS_NAME) $PS1"

if [ ! -d $TOP_DIR/.venv ]; then
    echo "no venv"
else
    rm -rf $TOP_DIR/.venv
fi

echo "Init .venv"
python3.8 -m venv $TOP_DIR/.venv
source $TOP_DIR/.venv/bin/activate
pip install -r $SCRIPT_DIR/requirements.txt
touch .done

export PS1=$CUSTOM_PS1


# Env variables
