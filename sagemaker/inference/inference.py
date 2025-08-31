"""
SageMaker PyTorch inference script for OpenCLIP.

Request (application/json):
{
  "task": "rank" | "encode_image" | "encode_text",
  "image_b64": "<base64-encoded image>",   # required for rank or encode_image
  "texts": ["a cat", "a dog"],             # required for rank or encode_text
  "top_k": 5                               # optional (rank)
}

Response (application/json):
- for task="rank": {"top_k": [{"text":"a cat","score":0.93}, ...]}
- for task="encode_image": {"image_embedding": [..]}
- for task="encode_text": {"text_embeddings": [[..], [..]]}
"""
from __future__ import annotations
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
import open_clip

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_image_from_b64(b64: str):
    byts = base64.b64decode(b64)
    return Image.open(io.BytesIO(byts)).convert("RGB")

# ---------- SageMaker model server hooks ----------

def model_fn(model_dir: str):
    """Load model and preprocessing artifacts from SM_MODEL_DIR."""
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.get("model_name", "ViT-B-32"),
        pretrained=None,  # we load weights from state_dict
        device=_DEVICE,
    )
    state = torch.load(model_dir / "open_clip_model.pt", map_location=_DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    tokenizer = open_clip.get_tokenizer(cfg.get("model_name", "ViT-B-32"))
    return {"model": model.to(_DEVICE), "preprocess": preprocess, "tokenizer": tokenizer, "cfg": cfg}

def input_fn(request_body: str, content_type: str = "application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.loads(request_body)

def predict_fn(inputs: Dict[str, Any], model_bundle, context=None):
    task = inputs.get("task", "rank")
    model = model_bundle["model"]
    preprocess = model_bundle["preprocess"]
    tokenizer = model_bundle["tokenizer"]

    with torch.no_grad(), torch.cuda.amp.autocast():
        if task == "encode_image":
            img = _load_image_from_b64(inputs["image_b64"])
            image = preprocess(img).unsqueeze(0).to(_DEVICE)
            feats = model.encode_image(image)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            return {"image_embedding": feats.squeeze(0).cpu().tolist()}

        if task == "encode_text":
            texts: List[str] = inputs["texts"]
            tokens = tokenizer(texts).to(_DEVICE)
            feats = model.encode_text(tokens)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            return {"text_embeddings": feats.cpu().tolist()}

        # default: rank texts for an image
        img = _load_image_from_b64(inputs["image_b64"])
        image = preprocess(img).unsqueeze(0).to(_DEVICE)
        texts = inputs["texts"]
        tokens = tokenizer(texts).to(_DEVICE)
        image_feats = torch.nn.functional.normalize(model.encode_image(image), dim=-1)
        text_feats = torch.nn.functional.normalize(model.encode_text(tokens), dim=-1)
        probs = (100.0 * image_feats @ text_feats.T).softmax(dim=-1).squeeze(0)
        top_k = min(int(inputs.get("top_k", 5)), len(texts))
        vals, idxs = torch.topk(probs, k=top_k)
        return {"top_k": [{"text": texts[i], "score": float(vals[j].item())} for j, i in enumerate(idxs.tolist())]}

def output_fn(prediction: Dict[str, Any], accept: str = "application/json"):
    return json.dumps(prediction), accept
