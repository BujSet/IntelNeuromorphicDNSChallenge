#!/bin/bash
#
# musdb_hq_snn_training.sh
# My CHTC job
#
export MPLCONFIGDIR=$(pwd)
export TORCH_EXTENSIONS_DIR=$(pwd)
export HOME=$(pwd)
rm -f docker_stderr
git clone https://github.com/BujSet/IntelNeuromorphicDNSChallenge.git -q
cd IntelNeuromorphicDNSChallenge/
export INTEL_NDNS_HOME=$(pwd)
git switch elise_branch -q
git submodule init -q
git submodule update -q
touch null_condor_config
export CONDOR_CONFIG=$(pwd)/null_condor_config
mv ~/musdb18hq.zip .
unzip -q musdb18hq.zip
python3 chtc_files/musdb_snn_training.py -path . "$@"
