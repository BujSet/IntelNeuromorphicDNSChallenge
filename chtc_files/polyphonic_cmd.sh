#!/bin/bash
export DATASET_DIR=/staging/groups/san_miguel_stacs_group/aam_dataset
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
mkdir -p aam_annotations/
mkdir -p aam_audio_mixes/
mkdir -p aam_audio_multitracks/
cd aam_annotations/
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 0001-1000] Initiating zipfile copy" \
        && cp $DATASET_DIR/0001-1000-annotations-v1.1.0.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 0001-1000] zipfile copy success" \
        && unzip 0001-1000-annotations-v1.1.0.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 0001-1000] Zipfile unpack success" \
        && rm -f 0001-1000-annotations-v1.1.0.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 0001-1000] Zipfile removal success" &
ANN1=$!
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 1001-2000] Initiating zipfile copy" \
        && cp $DATASET_DIR/1001-2000-annotations-v1.1.0.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 1001-2000] zipfile copy success" \
        && unzip 1001-2000-annotations-v1.1.0.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 1001-2000] Zipfile unpack success" \
        && rm -f 1001-2000-annotations-v1.1.0.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 1001-2000] Zipfile removal success" &
ANN2=$!
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 2001-3000] Initiating zipfile copy" \
        && cp $DATASET_DIR/2001-3000-annotations-v1.1.0.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 2001-3000] zipfile copy success" \
        && unzip 2001-3000-annotations-v1.1.0.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 2001-3000] Zipfile unpack success" \
        && rm -f 2001-3000-annotations-v1.1.0.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Annotations 2001-3000] Zipfile removal success" &
ANN3=$!
cd ../aam_audio_mixes/
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 0001-1000] Initiating zipfile copy" \
        && cp $DATASET_DIR/0001-1000-audio-mixes.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 0001-1000] zipfile copy success" \
        && unzip 0001-1000-audio-mixes.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 0001-1000] Zipfile unpack success" \
        && rm -f 0001-1000-audio-mixes.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 0001-1000] Zipfile removal success" &
MIX1=$!
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 1001-2000] Initiating zipfile copy" \
        && cp $DATASET_DIR/1001-2000-audio-mixes.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 1001-2000] zipfile copy success" \
        && unzip 1001-2000-audio-mixes.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 1001-2000] Zipfile unpack success" \
        && rm -f 1001-2000-audio-mixes.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 1001-2000] Zipfile removal success" &
MIX2=$!
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 2001-3000] Initiating zipfile copy" \
        && cp $DATASET_DIR/2001-3000-audio-mixes.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 2001-3000] zipfile copy success" \
        && unzip 2001-3000-audio-mixes.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 2001-3000] Zipfile unpack success" \
        && rm -f 2001-3000-audio-mixes.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Mixes 2001-3000] Zipfile removal success" &
MIX3=$!
cd ../aam_audio_multitracks/
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 0001-1000] Initiating zipfile copy" \
        && cp $DATASET_DIR/0001-1000-audio-multitracks.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 0001-1000] zipfile copy success" \
        && unzip 0001-1000-audio-multitracks.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 0001-1000] Zipfile unpack success" \
        && rm -f 0001-1000-audio-multitracks.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 0001-1000] Zipfile removal success" &
MUL1=$!
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 1001-2000] Initiating zipfile copy" \
        && cp $DATASET_DIR/1001-2000-audio-multitracks.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 1001-2000] zipfile copy success" \
        && unzip 1001-2000-audio-multitracks.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 1001-2000] Zipfile unpack success" \
        && rm -f 1001-2000-audio-multitracks.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 1001-2000] Zipfile removal success" &
MUL2=$!
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 2001-3000] Initiating zipfile copy" \
        && cp $DATASET_DIR/2001-3000-audio-multitracks.zip . \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 2001-3000] zipfile copy success" \
        && unzip 2001-3000-audio-multitracks.zip \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 2001-3000] Zipfile unpack success" \
        && rm -f 2001-3000-audio-multitracks.zip  \
	&& python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "[Tracks 2001-3000] Zipfile removal success" &
MUL3=$!
wait $ANN1 $ANN2 $ANN3 $MIX1 $MIX2 $MIX3 $MUL1 $MUL2 $MUL3 
python3 $INTEL_NDNS_HOME/chtc_files/ulog_script.py -msg "All data has been transferred, starting script" 
cd ../
# Need script to split data into training and validaiton
mkdir -p Trained/
# Need script to train network
#python3 other_models/pitch_snn.py -path ./ -epochs 50 -training_samples 8192 -validation_samples 8192 -is_CHTC_job -hiddenLayers $1 -b 64 -dataloader_workers 4 -dataloader_prefetch_factor 2 -saveCheckpoint
#mv ./Trained/pitch_snn_depth_*.pt ~/
