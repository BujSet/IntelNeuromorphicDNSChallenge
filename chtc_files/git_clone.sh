#!/bin/bash
GIT_START_TIME=$(date +%s)
git clone https://github.com/BujSet/IntelNeuromorphicDNSChallenge.git
cd IntelNeuromorphicDNSChallenge/
git switch develop
git submodule init
git submodule update
GIT_END_TIME=$(date +%s)
echo "Git operations took $((GIT_END_TIME - GIT_START_TIME)) seconds."
mkdir -p training_set/
mkdir -p validation_set/