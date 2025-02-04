#!/bin/bash
export TORCH_EXTENSIONS_DIR=$(pwd)
export HOME=$(pwd)
rm -f docker_stderr
git clone https://github.com/BujSet/IntelNeuromorphicDNSChallenge.git -q
cd IntelNeuromorphicDNSChallenge/
git switch develop -q
git submodule init -q
git submodule update -q
touch null_condor_config
export CONDOR_CONFIG=$(pwd)/null_condor_config
mkdir -p training_set/
mkdir -p validation_set/
cd training_set/
python3 ../chtc_files/ulog_script.py -msg "[Train Clean] Initiating tarball copy" \
        && cp /staging/groups/san_miguel_stacs_group/intel_speech_ndns_dataset/training_set/training_clean.tar.gz . \
	&& python3 ../chtc_files/ulog_script.py -msg "[Train Clean] Tarball copy success" \
        && tar -xzf training_clean.tar.gz \
	&& python3 ../chtc_files/ulog_script.py -msg "[Train Clean] Tarball unpack success" \
        && rm -f training_clean.tar.gz \
	&& python3 ../chtc_files/ulog_script.py -msg "[Train Clean] Tarball removal success" &
TCLEAN=$!
cd ../validation_set/
python3 ../chtc_files/ulog_script.py -msg "[Valid Clean] Initiating tarball copy" \
        && cp /staging/groups/san_miguel_stacs_group/intel_speech_ndns_dataset/validation_set/validation_clean.tar.gz . \
	&& python3 ../chtc_files/ulog_script.py -msg "[Valid Clean] Tarball copy success" \
        && tar -xzf validation_clean.tar.gz \
	&& python3 ../chtc_files/ulog_script.py -msg "[Valid Clean] Tarball unpack success" \
        && rm -f validation_clean.tar.gz \
	&& python3 ../chtc_files/ulog_script.py -msg "[Valid Clean] Tarball removal success" &
VCLEAN=$!
wait $TCLEAN $VCLEAN
cd ../
python3 chtc_files/ulog_script.py -msg "All data has been transferred, starting script" 
mkdir -p Trained/
python3 other_models/pitch_snn.py -path ./ -epochs 200 -training_samples 8192 -validation_samples 8192 -is_CHTC_job -hiddenLayerWidths 512 512 512 512 512 512 512 512 512 512 -hiddenLayers 10 -b 64 -dataloader_workers 4 -dataloader_prefetch_factor 2 -saveCheckpoint
mv ./Trained/pitch_snn_depth_10*.pt ~/
