FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt install -y python3.7 && apt install -y python3-pip
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3.7 -m pip install --upgrade pip
WORKDIR /app/
COPY requirements.txt .
RUN python3.7 -m pip install -r requirements.txt
# COPY src/ src/