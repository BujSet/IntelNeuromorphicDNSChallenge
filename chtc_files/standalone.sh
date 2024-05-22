#!/bin/bash
export TORCH_EXTENSIONS_DIR=$(pwd)
export HOME=$(pwd)
git clone https://github.com/BujSet/IntelNeuromorphicDNSChallenge.git -q
cd IntelNeuromorphicDNSChallenge/
git switch develop -q
git submodule init -q
git submodule update -q
mkdir -p training_set/
mkdir -p validation_set/
cd training_set/
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "[Train Clean] Initiating tarball copy" \
        && cp /staging/groups/lipasti_pharm_group/training_set/training_set_clean.tar.gz . \
        && echo "[Train Clean] Tarball copy success... Initiating tarball unpack" \
        && tar -xzf training_set_clean.tar.gz \
        && echo "[Train Clean] Tarball unpack success... Initiating tarball removal" \
        && rm -f training_set_clean.tar.gz \
        && echo "[Train Clean] Tarball removal success" &
TCLEAN=$!
echo "[Train Noise] Initiating tarball copy" \
        && cp /staging/groups/lipasti_pharm_group/training_set/training_set_noise.tar.gz . \
        && echo "[Train Noise] Tarball copy success... Initiating tarball unpack" \
        && tar -xzf training_set_noise.tar.gz \
        && echo "[Train Noise] Tarball unpack success... Initiating tarball removal" \
        && rm -f training_set_noise.tar.gz \
        && echo "[Train Noise] Tarball removal success" &
TNOISE=$!
echo "[Train Noisy] Initiating tarball copy" \
        && cp /staging/groups/lipasti_pharm_group/training_set/training_set_noisy.tar.gz . \
        && echo "[Train Noisy] Tarball copy success.. Initiating tarball unpack" \
        && tar -xzf training_set_noisy.tar.gz \
        && echo "[Train Noisy] Tarball unpack success... Initiating tarball removal" \
        && rm -f training_set_noisy.tar.gz \
        && echo "[Train Noisy] Tarball removal success" &
TNOISY=$!
cd ../validation_set/
echo "[Valid Clean] Initiating tarball copy" \
        && cp /staging/groups/lipasti_pharm_group/validation_set/validation_set_clean.tar.gz . \
        && echo "[Valid Clean] Tarball copy success... Initiating tarball unpack" \
        && tar -xzf validation_set_clean.tar.gz \
        && echo "[Valid Clean] Tarball unpack success... Initiating tarball removal" \
        && rm -f validation_set_clean.tar.gz \
        && echo "[Valid Clean] Tarball removal success" &
VCLEAN=$!
echo "[Valid Noise] Initiating tarball copy" \
        && cp /staging/groups/lipasti_pharm_group/validation_set/validation_set_noise.tar.gz . \
        && echo "[Valid Noise] Tarball copy success... Initiating tarball unpack" \
        && tar -xzf validation_set_noise.tar.gz \
        && echo "[Valid Noise] Tarball unpack success... Initiating tarball removal" \
        && rm -f validation_set_noise.tar.gz \
        && echo "[Valid Noise] Tarball removal success" &
VNOISE=$!
echo "[Valid Noisy] Initiating tarball copy" \
        && cp /staging/groups/lipasti_pharm_group/validation_set/validation_set_noisy.tar.gz . \
        && echo "[Valid Noisy] Tarball copy success... Initiating tarball unpack" \
        && tar -xzf validation_set_noisy.tar.gz \
        && echo "[Valid Noisy] Tarball unpack success... Initiating tarball removal" \
        && rm -f validation_set_noisy.tar.gz \
        && echo "[Valid Noisy] Tarball removal success" &
VNOISY=$!
wait $TCLEAN $TNOISE $TNOISY $VCLEAN $VNOISE $VNOISY
cd ../hrtfs/cipic/
echo "[CIPIC] Initiating tarball copy" \
        && cp /staging/groups/lipasti_pharm_group/cipic.tar.gz . \
        && echo "[CIPIC] Tarball copy success.. Initiating tarball unpack" \
        && tar -xzf cipic.tar.gz \
        && echo "[CIPIC] Tarball unpack success... Initiating tarball removal" \
        && rm -f cipic.tar.gz \
        && echo "[CIPIC] Tarball removal success" \
        && mv cipic/*.sofa . \
        && rm -rf cipic/ &
CIPIC_COPY=$!
wait $CIPIC_COPY
echo "$(date '+%Y-%m-%d %H:%M:%S')"
cd ../../
#nvidia-smi --query-gpu=timestamp,power.draw --format=csv  -i $NVIDIA_VISIBLE_DEVICES -lms 500 > ../power.txt &
#POWER_CNT=$!
#sleep 10s
#python other_models/binaural.py -training_epoch 15 -validation_epoch 1 -ssnns -cipicSubject 12 -spectrogram 2 -path ./ > timing_$1.txt
python other_models/cone.py -training_epoch 10 -path ./ > ../acc_5_cone.txt
#python other_models/train_sdnn.py -training_epoch 15 -validation_epoch 15 -ssnns -randomize_orients -path ./ > timing_$1.txt
#mv timing_$1.txt /staging/selagamsetty/
#mv training_orients.txt /staging/selagamsetty/training_orients_$1.txt
#mv validation_orients.txt /staging/selagamsetty/validation_orients_$1.txt
#mv initial_weights.png /staging/selagamsetty/initial_weights_$1.png
#mv final_weights.png /staging/selagamsetty/final_weights_$1.png
#mv Trained/network.pt /staging/selagamsetty/network_$1.pt
#sleep 10s
#kill ${POWER_CNT}
#python chtc_files/plot_power_profile.py
#mv power_profile.png ../
