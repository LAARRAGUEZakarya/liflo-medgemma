import runpod
import os
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
_proc_type = "unknown"


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
    global _proc_type

    # Attempt 1: Gemma3Processor.from_pretrained (direct, no auto-detection)
    try:
        from transformers import Gemma3Processor
        proc = Gemma3Processor.from_pretrained(MODEL_LOCAL)
        _proc_type = "Gemma3Processor"
        print(f"[processor] Loaded via {_proc_type}", flush=True)
        return proc
    except Exception as e:
        print(f"[processor] Gemma3Processor.from_pretrained failed: {e}", flush=True)

    # Attempt 2: PaliGemmaProcessor.from_pretrained
    try:
        from transformers import PaliGemmaProcessor
        proc = PaliGemmaProcessor.from_pretrained(MODEL_LOCAL)
        _proc_type = "PaliGemmaProcessor"
        print(f"[processor] Loaded via {_proc_type}", flush=True)
        return proc
    except Exception as e:
        print(f"[processor] PaliGemmaProcessor.from_pretrained failed: {e}", flush=True)

    # Attempt 3: manual construction
    print("[processor] Falling back to manual SiglipImageProcessor + AutoTokenizer", flush=True)
    from transformers import SiglipImageProcessor, AutoTokenizer, Gemma3Processor
    ip = SiglipImageProcessor.from_pretrained(MODEL_LOCAL)
    tk = AutoTokenizer.from_pretrained(MODEL_LOCAL)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    proc = Gemma3Processor(image_processor=ip, tokenizer=tk)
    if not getattr(proc, "chat_template", None) and getattr(tk, "chat_template", None):
        proc.chat_template = tk.chat_template
    _proc_type = "manual-Gemma3Processor"
    print(f"[processor] Loaded via {_proc_type}", flush=True)
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
    print(f"[model] Ready — device: {next(_model.parameters()).device} proc: {_proc_type}", flush=True)
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
            image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
            # Pass text + image DIRECTLY to processor — no apply_chat_template
            # This avoids the image_seq_length / token count mismatch that causes padding errors
            inputs = processor(
                images=[image],
                text=prompt,
                return_tensors="pt",
            ).to(model.device)
            print(f"[handler] image — input_ids: {inputs['input_ids'].shape} proc: {_proc_type}", flush=True)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text   = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
            print(f"[handler] text — input_ids: {inputs['input_ids'].shape}", flush=True)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        new_ids = output_ids[0][input_len:]
        result  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(f"[handler] done — in={input_len} out={len(new_ids)} result_len={len(result)}", flush=True)
        return {"report": result, "mode": mode}

    except Exception as e:
        print(f"[handler] ERROR:\n{traceback.format_exc()}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
