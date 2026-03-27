import runpod, os, base64, traceback
from io import BytesIO
import torch
torch.backends.cudnn.enabled = False

CACHE_DIR = "/runpod-volume/models"
MODEL_LOCAL = os.path.join(CACHE_DIR, "medgemma-4b-it")
MODEL_ID = "google/medgemma-4b-it"

_model = None
_processor = None


def download_model():
    from huggingface_hub import snapshot_download, login
    token = os.environ.get("HF_TOKEN")
    if not os.path.exists(os.path.join(MODEL_LOCAL, "config.json")):
        if token:
            login(token=token)
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_LOCAL)


def load_processor():
    from transformers import SiglipImageProcessor, AutoTokenizer
    image_processor = SiglipImageProcessor.from_pretrained(MODEL_LOCAL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL)
    try:
        from transformers import Gemma3Processor
        proc = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer)
    except Exception:
        from transformers import PaliGemmaProcessor
        proc = PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    # Copy chat template so apply_chat_template works on the manually-built processor
    if not getattr(proc, "chat_template", None) and getattr(tokenizer, "chat_template", None):
        proc.chat_template = tokenizer.chat_template
    return proc


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    download_model()
    _processor = load_processor()
    from transformers import Gemma3ForConditionalGeneration
    _model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_LOCAL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    _model.eval()
    return _model, _processor


def handler(job):
    job_input = job["input"]
    try:
        from PIL import Image
        model, processor = get_model()

        prompt   = str(job_input.get("prompt", "Analyse ce document médical."))
        mode     = str(job_input.get("mode", "chat"))
        img_b64  = job_input.get("image_base64", "")

        if img_b64:
            # ── IMAGE mode ──────────────────────────────────────────────────
            image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")

            # Gemma3 image format: {"type":"image"} token first, then text
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}]

            # Build the text with the image placeholder via apply_chat_template
            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: use tokenizer directly (older transformers)
                text = processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # Pass text as list to avoid padding issues with a single sample
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]
            # Image tokens can be 256–1024; cap new tokens so total < 1024
            safe_new = max(64, 1000 - input_len)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=safe_new,
                    do_sample=False          # greedy for stability
                )
            new_ids = output_ids[0][input_len:]
            result = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        else:
            # ── TEXT mode ───────────────────────────────────────────────────
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            # Gemma3 local-attention window = 1024 tokens total.
            # Keep max_new_tokens safely below that limit.
            safe_new = max(64, 1000 - input_len)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=safe_new,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            new_ids = output_ids[0][input_len:]
            result = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return {"report": result, "mode": mode}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
