#FROM ubuntu:20.04
#FROM nvcr.io/nvidia/pytorch:23.11-py3
#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime // maybe try using devel tag with later version
#FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
#FROM nvidia/cuda:10.1-base-ubuntu18.04
#FROM nvidia/cuda:12.3.1-devel-ubuntu20.04
#FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
#FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

LABEL org.opencontainers.image.authors="selagamsetty@wisc.edu"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
#RUN apt-get install -y software-properties-common

#RUN apt install -y ubuntu-drivers-common
#RUN apt install -y pciutils
#RUN apt-add-repository ppa:git-core/ppa
#RUN add-apt-repository ppa:graphics-drivers/ppa
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get update
RUN apt update 
RUN apt-get install --reinstall make
RUN apt-get install -y g++ libsndfile1
RUN apt install -y python3.10 git vim python3.10-venv
#RUN apt install -y python3.8 git vim python3.8-venv
RUN apt update 
RUN apt install -y emacs
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt install -y python3-pip pciutils wget

RUN python -m venv myenv
#python3.10-venv 
RUN python -m pip install h5py tensorboard soundfile htcondor torchaudio==2.1.0 python-sofa librosa==0.10.0 wheel pandas
#RUN python -m pip install h5py tensorboard soundfile htcondor torchaudio==2.0.2 python-sofa librosa==0.10.0 wheel pandas
#RUN python3 -m venv python3_venv
RUN python -m pip install pyroomacoustics
RUN python -m pip install -U pip
RUN python -m pip install https://github.com/lava-nc/lava-dl/releases/download/v0.5.0/lava_dl-0.5.0.tar.gz
RUN python -m pip install praat-parselmouth
#RUN apt-get install -y libsndfile1
#RUN python -m pip install https://github.com/lava-nc/lava-dl/releases/download/v0.3.2/lava_dl-0.3.2.tar.gz
#RUN python -m pip install torch==1.13.1
RUN python -m pip install --upgrade torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

#source ~/envs/myenv/bin/activatie


#RUN pip install https://github.com/lava-nc/lava-dl/releases/download/v0.5.0/lava-dl-0.5.0.tar.gz
#lava_nc-0.9.0.tar.gz  lava_dl-0.5.0.tar.gz 
#RUN apt install -y python3.7
#RUN apt install -y python3.10
#RUN ln -s /usr/bin/python3 /usr/bin/python
#RUN apt install -y curl
#RUN apt install -y python3.7-distutils
#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#RUN python3.7 get-pip.py
#RUN python -m pip install torch==1.13.1 torchvision=0.14.1 torchaudio
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2
#RUN apt-get install -y libsndfile1
#RUN apt install -y wget

#RUN pip install -r requirements.txt
#RUN python -c "import os; from distutils.sysconfig import get_python_lib; open(get_python_lib() + os.sep + 'ndns.pth', 'a').write(os.getcwd())"

#RUN apt-get update
#RUN 
#RUN pip install https://github.com/lava-nc/lava-dl/releases/download/v0.5.0/lava_dl-0.5.0.tar.gz 
#RUN python3.7 -m pip install https://github.com/lava-nc/lava-dl/releases/download/v0.3.2/lava_dl-0.3.2.tar.gz
#RUN pip install https://github.com/lava-nc/lava-dl/releases/download/v0.3.3/lava_dl-0.3.3.tar.gz

#RUN pip install cuda-python
#RUN pip install numpy==1.23.3 soundfile==0.11.0 librosa==0.9.2 configparser pandas torch==2.0.0
#RUN python3.7 -m pip install numpy soundfile==0.11.0 librosa==0.9.2 configparser pandas torch==1.13.1 matplotlib
#RUN pip install torch==2.0.0 onnx onnxruntime-gpu tensorboard soundfile
#RUN python3.7 -m pip install onnx onnxruntime-gpu tensorboard h5py
#RUN python -c "import os; from distutils.sysconfig import get_python_lib; open(get_python_lib() + os.sep + 'ndns.pth', 'a').write(os.getcwd())"
# Cleaning up OS package manager
#RUN apt-get clean -y
#RUN apt-get autoremove -y
#RUN rm -rf /var/lib/apt/lists/*
