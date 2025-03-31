#!/bin/bash
#
# hybrid_demucs.sh
# My CHTC job
#
export MPLCONFIGDIR=$(pwd)
export TORCH_EXTENSIONS_DIR=$(pwd)
export HOME=$(pwd)
python3 hybrid_demucs_tutorial.py #
