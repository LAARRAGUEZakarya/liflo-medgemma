FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN pip install --upgrade pip && \
    pip install "transformers>=4.51.0" accelerate Pillow runpod \
                sentencepiece protobuf huggingface_hub

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
