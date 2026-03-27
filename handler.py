import runpod
import os
import json
import base64
import traceback
from io import BytesIO
import torch

torch.backends.cudnn.enabled = False

CACHE_DIR   = "/runpod-volume/models"
MODEL_LOCAL = os.path.join(CACHE_DIR, "medgemma-4b-it")
MODEL_ID    = "google/medgemma-4b-it"

_model     = None
_processor = None


def download_model():
    from huggingface_hub import snapshot_download, login
    token = os.environ.get("HF_TOKEN")
    if not os.path.exists(os.path.join(MODEL_LOCAL, "config.json")):
        print("Downloading MedGemma...", flush=True)
        if token:
            login(token=token)
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_LOCAL)
        print("Download complete.", flush=True)


def load_processor():
    from transformers import SiglipImageProcessor, AutoTokenizer

    print("[processor] Loading SiglipImageProcessor...", flush=True)
    image_processor = SiglipImageProcessor.from_pretrained(MODEL_LOCAL)

    print("[processor] Loading AutoTokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL)

    try:
        from transformers import Gemma3Processor
        proc = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer)
        print("[processor] Using Gemma3Processor", flush=True)
    except Exception as e:
        print(f"[processor] Gemma3Processor failed ({e}), trying PaliGemmaProcessor...", flush=True)
        from transformers import PaliGemmaProcessor
        proc = PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        print("[processor] Using PaliGemmaProcessor", flush=True)

    return proc


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    download_model()
    _processor = load_processor()

    print("[model] Loading Gemma3ForConditionalGeneration...", flush=True)
    from transformers import Gemma3ForConditionalGeneration
    _model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_LOCAL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    print(f"[model] Ready on: {next(_model.parameters()).device}", flush=True)
    return _model, _processor


def handler(job):
    job_input = job["input"]
    try:
        from PIL import Image

        model, processor = get_model()

        prompt  = str(job_input.get("prompt", "Analyse ce document médical et décris ce que tu observes."))
        mode    = str(job_input.get("mode", "chat"))
        img_b64 = job_input.get("image_base64", "")

        if img_b64:
            image    = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            # use tokenizer.apply_chat_template — processor built manually has no chat_template
            text   = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text   = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        new_ids = output_ids[0][input_len:]
        result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(f"[handler] mode={mode} in={input_len} out={len(new_ids)}", flush=True)
        return {"report": result, "mode": mode}

    except Exception as e:
        print(f"[handler] ERROR:\n{traceback.format_exc()}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
