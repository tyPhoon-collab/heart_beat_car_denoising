
# Choose docker image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
#FROM tensorflow/tensorflow:1.15.0-gpu-py3
#FROM python:3.9

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# NVIDIAの公開鍵を設定
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y gnupg

# 時間指定
# ENV TZ=Asia/Tokyo
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
