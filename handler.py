import runpod, os, base64, traceback, sys
from io import BytesIO
import torch
torch.backends.cudnn.enabled = False

CACHE_DIR   = "/runpod-volume/models"
MODEL_LOCAL = os.path.join(CACHE_DIR, "medgemma-4b-it")
MODEL_ID    = "google/medgemma-4b-it"
IMAGE_SIZE  = 896

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
        log("Model already on disk.")


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    download_model()

    log("Loading AutoProcessor …")
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    _processor = AutoProcessor.from_pretrained(MODEL_LOCAL)

    # Disable pan-and-scan at the object level so the image processor never
    # produces variable-size crops (which cannot be stacked into one tensor).
    if hasattr(_processor, "image_processor"):
        _processor.image_processor.do_pan_and_scan = False
        log("pan-and-scan disabled on image_processor.")

    log("Loading model …")
    _model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_LOCAL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    log("Model ready.")
    return _model, _processor


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
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            log(f"Image resized to {IMAGE_SIZE}x{IMAGE_SIZE}")

            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]

            log("Applying chat template …")
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Full processor call: handles image-token expansion in input_ids
            # numpy<2.0 in Dockerfile ensures torch.from_numpy() works correctly
            log("Processing text + image …")
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,   
                max_length=900,
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]
            safe_new  = max(32, 1024 - input_len)

            log(f"Generating (input_len={input_len}, max_new={safe_new}) …")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=safe_new,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            new_ids = output_ids[0][input_len:]
            result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            log(f"Done. {len(result)} chars.")

        else:
            # ── TEXT mode ────────────────────────────────────────────────────
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            log("Applying chat template (text only) …")
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            log("Tokenising …")
            inputs    = processor.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,  
                max_length=900,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            safe_new  = max(32, 1024 - input_len)

            log(f"Generating (input_len={input_len}, max_new={safe_new}) …")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=safe_new,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            new_ids = output_ids[0][input_len:]
            result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            log(f"Done. {len(result)} chars.")

        return {"report": result, "mode": mode}

    except Exception as e:
        log(f"ERROR: {e}")
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
