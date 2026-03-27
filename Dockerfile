FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "transformers>=4.51.0" accelerate Pillow runpod \
                sentencepiece protobuf huggingface_hub && \
    pip install torch==2.1.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
        --force-reinstall

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
