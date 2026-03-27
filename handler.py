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


def patch_configs(model_dir):
    """Patch local config files so transformers recognises the processor."""
    # preprocessor_config.json — add image_processor_type if missing
    prep_path = os.path.join(model_dir, "preprocessor_config.json")
    if os.path.exists(prep_path):
        with open(prep_path) as f:
            prep = json.load(f)
        print(f"[patch] preprocessor_config keys: {list(prep.keys())}", flush=True)
        if "image_processor_type" not in prep:
            prep["image_processor_type"] = "Gemma3ImageProcessor"
            with open(prep_path, "w") as f:
                json.dump(prep, f, indent=2)
            print("[patch] added image_processor_type = Gemma3ImageProcessor", flush=True)
    else:
        print("[patch] WARNING: preprocessor_config.json not found", flush=True)

    # config.json — log model_type for debugging
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"[patch] config.json model_type: {cfg.get('model_type')}", flush=True)


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from huggingface_hub import snapshot_download, login

    token = os.environ.get("HF_TOKEN")

    if not os.path.exists(os.path.join(MODEL_LOCAL, "config.json")):
        print("Downloading MedGemma...", flush=True)
        if token:
            login(token=token)
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_LOCAL)
        print("Download complete.", flush=True)

    patch_configs(MODEL_LOCAL)

    print("Loading processor...", flush=True)
    _processor = AutoProcessor.from_pretrained(MODEL_LOCAL)

    print("Loading model...", flush=True)
    _model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_LOCAL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    print(f"MedGemma ready on: {next(_model.parameters()).device}", flush=True)
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
            text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        new_ids = output_ids[0][input_len:]
        result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        print(f"[handler] mode={mode} in={input_len} out={len(new_ids)}", flush=True)
        return {"report": result, "mode": mode}

    except Exception as e:
        print(f"[handler] ERROR: {traceback.format_exc()}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
