FROM nvidia/cuda:11.4-cudnn8-devel-ubuntu:18.04
ENV CUDA_PATH /usr/local/cuda
COPY . /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.6 \
    python3-pip
RUN pip3 install --upgrade "pip < 21.1.3"
RUN pip install tensorflow==2.2
RUN pip install tensorflow-addons==0.10.0
RUN pip install scikit-image==0.17.2 tqdm==4.61.0 tensorflow-gpu=2.2 
RUN pip install numpy==1.19.5 matplotlib==3.3.4
RUN pip oyaml==1.0

