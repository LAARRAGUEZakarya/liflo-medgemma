import runpod
import os
import base64
import traceback
from io import BytesIO

CACHE_DIR = "/runpod-volume/models"

_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    MODEL_ID = "google/medgemma-4b-it"
    token = os.environ.get("HF_TOKEN")

    print("Loading MedGemma processor...")
    _processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=token,
        cache_dir=CACHE_DIR,
    )

    print("Loading MedGemma model...")
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
        cache_dir=CACHE_DIR,
    )
    _model.eval()
    print("MedGemma ready.")
    return _model, _processor


def handler(job):
    job_input = job["input"]
    try:
        import torch
        from PIL import Image

        model, processor = get_model()

        prompt   = str(job_input.get("prompt", "Analyse ce document médical."))
        mode     = str(job_input.get("mode", "chat"))
        img_b64  = job_input.get("image_base64", "")

        if img_b64:
            image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt},
            ]}]
            inputs = processor(
                images=[image],
                text=processor.apply_chat_template(messages, add_generation_prompt=True),
                return_tensors="pt",
            ).to(model.device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = processor(
                text=processor.apply_chat_template(messages, add_generation_prompt=True),
                return_tensors="pt",
            ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        result = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        return {"report": result.strip(), "mode": mode}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
