#!/bin/bash
#
# musdb_snn_cipic_10465856_2_save_audio.sh
# My CHTC job
#
# Loads the network.pt checkpoint that musdb_snn_cipic_10465856_2_epochs50
# (see musdb_snn_cipic_10465856_2_epochs50/args.txt) already trained, and
# reuses musdb_hq_snn_training.sh's setup steps unchanged except for moving
# that checkpoint into place. The .sub file passes -epochs 0 -useCheckpoint
# so no further training happens, and -save_audio_dir so the test-set
# separated vocals/accompaniment audio gets written out instead of only
# scored.
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
mv ~/network.pt .

if [[ " $* " == *" -useCipic "* ]]; then
    cd hrtfs/cipic/
    echo "[CIPIC] Initiating tarball copy" \
            && mv ~/cipic_dataset.tar.gz . \
            && echo "[CIPIC] Tarball copy success.. Initiating tarball unpack" \
            && tar -xzf cipic_dataset.tar.gz \
            && echo "[CIPIC] Tarball unpack success... Initiating tarball removal" \
            && rm -f cipic_dataset.tar.gz \
            && echo "[CIPIC] Tarball removal success" \
            && mv cipic/*.sofa . \
            && rm -rf cipic/
    cd ../../
fi

python3 chtc_files/musdb_snn_training.py -path . "$@"
