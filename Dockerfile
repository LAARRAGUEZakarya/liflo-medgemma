FROM runpod/base:0.4.0-cuda11.8.0

RUN apt-get update && apt-get install -y git python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install transformers accelerate Pillow runpod

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
