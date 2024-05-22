#!/bin/bash
SHARED_DIR="/staging/groups/lipasti_pharm_group"
PROJ_HOME="~/IntelNeuromorphicDNSChallenge"

# Navigate to where the desired destination
cd ${PROJ_HOME}/hrtfs/cipic/

echo "[CIPIC] Initiating tarball copy"
cp ${SHARED_DIR}/cipic.tar.gz .

echo "[CIPIC] Tarball copy success.. Initiating tarball unpack"
tar -xzf cipic.tar.gz

echo "[CIPIC] Tarball unpack success... Initiating tarball removal"
rm -f cipic.tar.gz

echo "[CIPIC] Tarball removal success"
mv cipic/*.sofa .

# Remove unecessary nested dir that should now be empty
rm -rf cipic/
