FROM nvidia/cuda:12.1.1-devel-ubuntu20.04
MAINTAINER selagamsetty@wisc.edu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt update
RUN apt-get install --reinstall make
RUN apt-get install -y g++ libsndfile1
RUN apt install -y python3.8 git vim python3.8-venv
RUN apt update
RUN apt install -y emacs
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt install -y python3-pip pciutils wget
RUN python -m venv myenv
RUN python -m pip install h5py tensorboard soundfile htcondor torchaudio==2.0.2 python-sofa librosa==0.10.0 wheel pandas
RUN python -m pip install pyroomacoustics
RUN python -m pip install -U pip
RUN python -m pip install https://github.com/lava-nc/lava-dl/releases/download/v0.5.0/lava_dl-0.5.0.tar.gz
