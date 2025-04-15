#!/bin/bash
#
# hybrid_demucs.sh
#
export MPLCONFIGDIR=$(pwd)
export TORCH_EXTENSIONS_DIR=$(pwd)
export HOME=$(pwd)
git clone https://github.com/BujSet/IntelNeuromorphicDNSChallenge.git -q
cd IntelNeuromorphicDNSChallenge/
export INTEL_NDNS_HOME=$(pwd)
git switch elise_branch -q
git submodule init -q
git submodule update -q
touch null_condor_config
export CONDOR_CONFIG=$(pwd)/null_condor_config
mv ~/musdb18hq.zip .
unzip musdb18hq.zip
python3 chtc_files/hybrid_demucs_full_dataset.py test
