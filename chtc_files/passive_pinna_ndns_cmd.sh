#!/bin/bash
export TORCH_EXTENSIONS_DIR=$(pwd)
export HOME=$(pwd)
git clone https://github.com/BujSet/IntelNeuromorphicDNSChallenge.git -q
cd IntelNeuromorphicDNSChallenge/
python3 chtc_files/ulog_script.py -msg "Test Message"
export INTEL_NDNS_HOME=$(pwd)
git switch develop -q
git submodule init -q
git submodule update -q
mkdir -p validation_set/
echo "$(date '+%Y-%m-%d %H:%M:%S')"
cd validation_set/
condor_chirp ulog "Default condor_chirp command"
/usr/libexec/condor_chirp ulog "condor_chirp command from /usr dir"
python3 ../chtc_files/ulog_script.py -msg "Initiating dataset copy from staging to execution point"
echo "[Valid Clean] Initiating tarball copy" \
        && cp /staging/groups/san_miguel_stacs_group/intel_speech_ndns_dataset/validation_set/validation_clean.tar.gz . \
	&& echo "[Valid Clean] Tarball copy success... Initiating tarball unpack" \
        && tar -xzf validation_clean.tar.gz \
	&& echo "[Valid Clean] Tarball unpack success... Initiating tarball removal" \
        && rm -f validation_clean.tar.gz \
	&& echo "[Valid Clean] Tarball removal success" &
VCLEAN=$!
echo "[Valid Noise] Initiating tarball copy" \
        && cp /staging/groups/san_miguel_stacs_group/intel_speech_ndns_dataset/validation_set/validation_noise.tar.gz . \
	&& echo "[Valid Noise] Tarball copy success... Initiating tarball unpack" \
        && tar -xzf validation_noise.tar.gz \
	&& echo "[Valid Noise] Tarball unpack success... Initiating tarball removal" \
        && rm -f validation_noise.tar.gz \
	&& echo "[Valid Noise] Tarball removal success" &
VNOISE=$!
wait $VCLEAN $VNOISE
pushd $INTEL_NDNS_HOME
python3 chtc_files/ulog_script.py -msg "Completed dataset copy from staging to execution point"
popd
cd ../hrtfs/cipic/
echo "[CIPIC] Initiating tarball copy" \
        && cp /staging/groups/san_miguel_stacs_group/cipic_dataset/cipic_dataset.tar.gz . \
	&& echo "[CIPIC] Tarball copy success.. Initiating tarball unpack" \
        && tar -xzf cipic_dataset.tar.gz \
	&& echo "[CIPIC] Tarball unpack success... Initiating tarball removal" \
        && rm -f cipic_dataset.tar.gz \
	&& echo "[CIPIC] Tarball removal success" \
        && mv cipic/*.sofa . \
        && rm -rf cipic/ &
CIPIC_COPY=$!
wait $CIPIC_COPY
echo "$(date '+%Y-%m-%d %H:%M:%S')"
cd ../../
if [ "$9" -eq 1 ]; then 
    python3 other_models/validate_si_snr.py -cipicSubject $1 -cipicChannel $2 -speechFilterOrient $3 -noiseFilterOrientStart $4 -noiseFilterOrientEnd $5 -b $6 -dataloader_workers $7 -dataloader_prefetch_factor $8 -path ./ -print_validation_results_header -is_CHTC_job -save_profile_trace
else
    python3 other_models/validate_si_snr.py -cipicSubject $1 -cipicChannel $2 -speechFilterOrient $3 -noiseFilterOrientStart $4 -noiseFilterOrientEnd $5 -b $6 -dataloader_workers $7 -dataloader_prefetch_factor $8 -path ./ -print_validation_results_header -is_CHTC_job
fi

