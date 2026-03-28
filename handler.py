import runpod, os, base64, traceback, sys
from io import BytesIO
import torch
torch.backends.cudnn.enabled = False

CACHE_DIR   = "/runpod-volume/models"
MODEL_LOCAL = os.path.join(CACHE_DIR, "medgemma-4b-it")
MODEL_ID    = "google/medgemma-4b-it"
IMAGE_SIZE  = 896          # MedGemma / SigLIP native resolution

_model     = None
_processor = None


def log(msg):
    print(f"[HANDLER] {msg}", file=sys.stderr, flush=True)


def download_model():
    from huggingface_hub import snapshot_download, login
    token = os.environ.get("HF_TOKEN")
    if not os.path.exists(os.path.join(MODEL_LOCAL, "config.json")):
        log("Downloading model from HuggingFace …")
        if token:
            login(token=token)
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_LOCAL)
    else:
        log("Model already on disk, skipping download.")


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    download_model()

    log("Loading AutoProcessor …")
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    _processor = AutoProcessor.from_pretrained(MODEL_LOCAL)
    log("Processor loaded.")

    log("Loading Gemma3ForConditionalGeneration …")
    _model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_LOCAL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    _model.eval()
    log("Model loaded and ready.")
    return _model, _processor


def pil_to_pixel_values(pil_image):
    """
    PIL RGB image → (1, 3, H, W) bfloat16 tensor.

    Uses ONLY PIL + pure PyTorch (torch.frombuffer).
    No numpy, no torchvision — avoids 'Numpy is not available' entirely.
    """
    from PIL import Image
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS).convert("RGB")

    raw  = bytearray(pil_image.tobytes())                    # H*W*3 raw uint8 bytes
    flat = torch.frombuffer(raw, dtype=torch.uint8).clone()  # (H*W*3,) — clone to own memory
    t    = flat.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)           # (H, W, 3)
    t    = t.float().div_(255.0)                             # [0, 1]
    t    = t.permute(2, 0, 1)                                # (3, H, W)
    t    = t.sub_(0.5).div_(0.5)                             # [-1, 1]  (SigLIP norm)
    return t.unsqueeze(0).to(dtype=torch.bfloat16)           # (1, 3, H, W)


def handler(job):
    job_input = job["input"]
    try:
        from PIL import Image
        model, processor = get_model()

        prompt  = str(job_input.get("prompt", "Analyse ce document médical."))
        mode    = str(job_input.get("mode", "chat"))
        img_b64 = job_input.get("image_base64", "")

        log(f"Job mode: {'image' if img_b64 else 'text'}")

        if img_b64:
            # ── IMAGE mode ───────────────────────────────────────────────────
            image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
            log(f"Image decoded: {image.size}")

            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]

            log("Applying chat template …")
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            log("Tokenising …")
            text_inputs = processor.tokenizer(
                text, return_tensors="pt", padding=True
            ).to(model.device)

            log("Building pixel_values …")
            pixel_values = pil_to_pixel_values(image).to(model.device)

            inputs    = {**text_inputs, "pixel_values": pixel_values}
            input_len = text_inputs["input_ids"].shape[-1]
            safe_new  = max(64, 1000 - input_len)

            log(f"Generating (max_new_tokens={safe_new}) …")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=safe_new,
                    do_sample=False,
                )

            new_ids = output_ids[0][input_len:]
            result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            log(f"Done. Output length: {len(result)} chars")

        else:
            # ── TEXT mode ────────────────────────────────────────────────────
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            log("Applying chat template (text only) …")
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            log("Tokenising …")
            inputs    = processor.tokenizer(
                text, return_tensors="pt", padding=True
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            safe_new  = max(64, 1000 - input_len)

            log(f"Generating (max_new_tokens={safe_new}) …")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=safe_new,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            new_ids = output_ids[0][input_len:]
            result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            log(f"Done. Output length: {len(result)} chars")

        return {"report": result, "mode": mode}

    except Exception as e:
        log(f"ERROR: {e}")
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
