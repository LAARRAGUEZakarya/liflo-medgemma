FROM runpod/base:0.4.0-cuda11.8.0

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install transformers accelerate Pillow runpod

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
