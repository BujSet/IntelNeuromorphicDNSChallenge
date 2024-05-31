#!/bin/bash
SHARED_DIR="/staging/groups/lipasti_pharm_group"
export TORCH_EXTENSIONS_DIR=$(pwd)
export HOME=$(pwd)
GIT_START_TIME=$(date +%s)
git clone https://github.com/BujSet/IntelNeuromorphicDNSChallenge.git
cd IntelNeuromorphicDNSChallenge/
git switch develop
git submodule init
git submodule update
git switch rob_dev

GIT_END_TIME=$(date +%s)
echo "Git operations took $((GIT_END_TIME - GIT_START_TIME)) seconds."

mkdir -p training_set/
mkdir -p validation_set/
mkdir -p models/
cd training_set/

TARBALL_START_TIME=$(date +%s)
echo "[Train Clean] Initiating tarball copy" \
		&& cp ${SHARED_DIR}/training_set/training_set_clean.tar.gz . \
	&& echo "[Train Clean] Tarball copy success" \
	&& echo "[Train Clean] Initiating tarball unpack" \
		&& tar -xzf training_set_clean.tar.gz \
	&& echo "[Train Clean] Tarball unpack success" \
	&& echo "[Train Clean] Initiating tarball removal" \
		&& rm -f training_set_clean.tar.gz \
	&& echo "[Train Clean] Tarball removal success" &
TCLEAN=$!
echo "[Train Noise] Initiating tarball copy" \
		&& cp ${SHARED_DIR}/training_set/training_set_noise.tar.gz . \
	&& echo "[Train Noise] Tarball copy success" \
	&& echo "[Train Noise] Initiating tarball unpack" \
		&& tar -xzf training_set_noise.tar.gz \
	&& echo "[Train Noise] Tarball unpack success" \
	&& echo "[Train Noise] Initiating tarball removal" \
		&& rm -f training_set_noise.tar.gz \
	&& echo "[Train Noise] Tarball removal success" &
TNOISE=$!
echo "[Train Noisy] Initiating tarball copy" \
        && cp ${SHARED_DIR}/training_set/training_set_noisy.tar.gz . \
	&& echo "[Train Noisy] Tarball copy success" \
	&& echo "[Train Noisy] Initiating tarball unpack" \
        && tar -xzf training_set_noisy.tar.gz \
	&& echo "[Train Noisy] Tarball unpack success" \
	&& echo "[Train Noisy] Initiating tarball removal" \
        && rm -f training_set_noisy.tar.gz \
	&& echo "[Train Noisy] Tarball removal success" &
TNOISY=$!
cd ../validation_set/
echo "[Valid Clean] Initiating tarball copy" \
  && cp ${SHARED_DIR}/validation_set/validation_set_clean.tar.gz . \
	&& echo "[Valid Clean] Tarball copy success" \
	&& echo "[Valid Clean] Initiating tarball unpack" \
        && tar -xzf validation_set_clean.tar.gz \
	&& echo "[Valid Clean] Tarball unpack success" \
	&& echo "[Valid Clean] Initiating tarball removal" \
        && rm -f validation_set_clean.tar.gz \
	&& echo "[Valid Clean] Tarball removal success" &
VCLEAN=$!
echo "[Valid Noise] Initiating tarball copy" \
        && cp ${SHARED_DIR}/validation_set/validation_set_noise.tar.gz . \
	&& echo "[Valid Noise] Tarball copy success" \
	&& echo "[Valid Noise] Initiating tarball unpack" \
        && tar -xzf validation_set_noise.tar.gz \
	&& echo "[Valid Noise] Tarball unpack success" \
	&& echo "[Valid Noise] Initiating tarball removal" \
        && rm -f validation_set_noise.tar.gz \
	&& echo "[Valid Noise] Tarball removal success" &
VNOISE=$!
echo "[Valid Noisy] Initiating tarball copy" \
        && cp ${SHARED_DIR}/validation_set/validation_set_noisy.tar.gz . \
	&& echo "[Valid Noisy] Tarball copy success" \
	&& echo "[Valid Noisy] Initiating tarball unpack" \
        && tar -xzf validation_set_noisy.tar.gz \
	&& echo "[Valid Noisy] Tarball unpack success" \
	&& echo "[Valid Noisy] Initiating tarball removal" \
        && rm -f validation_set_noisy.tar.gz \
	&& echo "[Valid Noisy] Tarball removal success" &
VNOISY=$!
wait $TCLEAN $TNOISE $TNOISY $VCLEAN $VNOISE $VNOISY
cd ../hrtfs/cipic/
TARBALL_END_TIME=$(date +%s)
echo "Tarball operations took $((TARBALL_END_TIME - TARBALL_START_TIME)) seconds."

DOWNLOAD_CIPIC_START_TIME=$(date +%s)

echo "[CIPIC HRTFs] Downloading HRTFs from CIPIC database" \
        && ./download.sh \
        && echo "[CIPIC HRTFs] Filter Dataset download success"
cd ../../
DOWNLOAD_CIPIC_END_TIME=$(date +%s)
echo "CIPIC download operations took $((DOWNLOAD_CIPIC_END_TIME - DOWNLOAD_CIPIC_START_TIME)) seconds."

# nvidia-smi --query-gpu=timestamp,power.draw --format=csv  -i $(python chtc_files/get_device_uuid.py) -lms 500 > ../power.txt &
# POWER_CNT=$!
# sleep 10s
echo "Starting main Python script"
PYTHON_START_TIME=$(date +%s)
python fft-cnn/model/train-chtc.py -epoch 1 -path ./ > ../accuracy-fft-cnn.txt
PYTHON_END_TIME=$(date +%s)
echo "Main Python script execution took $((PYTHON_END_TIME - PYTHON_START_TIME)) seconds."

# sleep 10s
# kill ${POWER_CNT}
# python chtc_files/plot_power_profile.py
# mv power_profile.png ../
