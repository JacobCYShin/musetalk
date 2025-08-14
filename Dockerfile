FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

WORKDIR /workspace

# 1) 시스템 패키지 + pip + PyTorch 한 번에 설치
RUN apt-get update && \
    apt-get install -y git wget curl unzip ffmpeg build-essential \
                       python3 python3-pip python3-venv python3-dev && \
    python3 -m pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
  
RUN pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html