import runpod
import os
import base64
import traceback
from io import BytesIO
import torch

torch.backends.cudnn.enabled = False  # same as your working pod

CACHE_DIR   = "/runpod-volume/models"
MODEL_LOCAL = os.path.join(CACHE_DIR, "medgemma-4b-it")
MODEL_ID    = "google/medgemma-4b-it"

_model     = None
_processor = None


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from huggingface_hub import snapshot_download, login

    token = os.environ.get("HF_TOKEN")

    # Download to network volume if not already there (same as your start.sh)
    if not os.path.exists(os.path.join(MODEL_LOCAL, "config.json")):
        print("Downloading MedGemma to network volume...", flush=True)
        if token:
            login(token=token)
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_LOCAL)
        print("Download complete.", flush=True)

    # Load from LOCAL path — this is what your pod did and it worked
    print("Loading processor from local path...", flush=True)
    _processor = AutoProcessor.from_pretrained(MODEL_LOCAL)

    print("Loading model from local path...", flush=True)
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
            # Image mode — same as /medgemma-image in your pod API
            image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
        else:
            # Text-only mode — same as /medgemma in your pod API
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        new_ids = output_ids[0][input_len:]
        result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        print(f"[handler] mode={mode} input_len={input_len} output_len={len(new_ids)}", flush=True)
        return {"report": result, "mode": mode}

    except Exception as e:
        print(f"[handler] ERROR: {traceback.format_exc()}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
