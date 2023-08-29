# This is reference!
#!/bin/bash

source ~/.bashrc

SCRIPT_DIR=$(realpath $(dirname $BASH_SOURCE))
TOP_DIR=$(realpath $SCRIPT_DIR/..)
PS_NAME=$(basename $TOP_DIR)
CUSTOM_PS1="($PS_NAME) $PS1"

if [ ! -d $TOP_DIR/.venv ]; then
    echo "Init .venv"
    virtualenv -p /usr/bin/python3 $TOP_DIR/.venv
    source $TOP_DIR/.venv/bin/activate
    pip install -r $SCRIPT_DIR/requirements.txt
else
    source $TOP_DIR/.venv/bin/activate
fi
export PS1=$CUSTOM_PS1

# Env variables
export NPU_SIM_TOP_DIR=$TOP_DIR
