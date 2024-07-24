FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
WORKDIR /workspace/StoryDiffusion
ENV CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt install -y git

COPY requirements.txt /workspace/StoryDiffusion
RUN cd StoryDiffusion && pip install -r . requirements.txt

COPY . /workspace/StoryDiffusion
