import runpod, os, base64, traceback
from io import BytesIO
import torch
import torchvision.transforms.functional as TF
torch.backends.cudnn.enabled = False

CACHE_DIR  = "/runpod-volume/models"
MODEL_LOCAL = os.path.join(CACHE_DIR, "medgemma-4b-it")
MODEL_ID    = "google/medgemma-4b-it"

# MedGemma / SigLIP native resolution and normalisation
IMAGE_SIZE  = 896
IMG_MEAN    = [0.5, 0.5, 0.5]
IMG_STD     = [0.5, 0.5, 0.5]

_model     = None
_processor = None


def download_model():
    from huggingface_hub import snapshot_download, login
    token = os.environ.get("HF_TOKEN")
    if not os.path.exists(os.path.join(MODEL_LOCAL, "config.json")):
        if token:
            login(token=token)
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_LOCAL)


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    download_model()

    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    _processor = AutoProcessor.from_pretrained(MODEL_LOCAL)

    _model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_LOCAL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    _model.eval()
    return _model, _processor


def pil_to_pixel_values(pil_image: "PIL.Image.Image") -> torch.Tensor:
    """
    Convert a PIL image to a (1, 3, H, W) bfloat16 tensor using torchvision only.
    Completely avoids the numpy dependency that crashes Gemma3ImageProcessor.
    """
    from PIL import Image
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    # TF.to_tensor: PIL RGB → float32 [0,1] tensor (C,H,W), pure torch — no numpy
    t = TF.to_tensor(pil_image)
    t = TF.normalize(t, mean=IMG_MEAN, std=IMG_STD)       # → [-1, 1]
    return t.unsqueeze(0).to(dtype=torch.bfloat16)        # → (1, 3, 896, 896)


def handler(job):
    job_input = job["input"]
    try:
        from PIL import Image
        model, processor = get_model()

        prompt  = str(job_input.get("prompt", "Analyse ce document médical."))
        mode    = str(job_input.get("mode", "chat"))
        img_b64 = job_input.get("image_base64", "")

        if img_b64:
            # ── IMAGE mode ───────────────────────────────────────────────────
            image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")

            # Build chat template — inserts <image> placeholder token
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenise text — the <image> special token is kept as-is;
            # the model expands it to image patch embeddings at forward time
            text_inputs = processor.tokenizer(
                text, return_tensors="pt", padding=True
            ).to(model.device)

            # Build pixel_values with torchvision (zero numpy dependency)
            pixel_values = pil_to_pixel_values(image).to(model.device)

            inputs    = {**text_inputs, "pixel_values": pixel_values}
            input_len = text_inputs["input_ids"].shape[-1]
            safe_new  = max(64, 1000 - input_len)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=safe_new,
                    do_sample=False,
                )

            new_ids = output_ids[0][input_len:]
            result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        else:
            # ── TEXT mode ────────────────────────────────────────────────────
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs    = processor.tokenizer(
                text, return_tensors="pt", padding=True
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            safe_new  = max(64, 1000 - input_len)

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

        return {"report": result, "mode": mode}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
