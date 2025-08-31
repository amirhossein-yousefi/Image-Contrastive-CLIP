from typing import List, Tuple, Any
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

def get_model_device_dtype(model) -> Tuple[torch.device, torch.dtype]:
    """
    Return the device and dtype of a model by inspecting the first parameter.
    """
    p = next(model.parameters())
    return p.device, p.dtype

def _device_is_cuda(device: Any) -> bool:
    """
    Robust check whether the given device is CUDA.
    Accepts torch.device or str.
    """
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.startswith("cuda")
    return False

def embed_images(
    model, processor, images: List[Any], device: torch.device, batch_size: int = 128
) -> torch.Tensor:
    """
    Compute L2-normalized image embeddings using model.get_image_features.
    Returns a float32 tensor on CPU/GPU with shape (N, D).
    Defaults preserved: batch_size=128.
    """
    model.eval()
    _, model_dtype = get_model_device_dtype(model)
    feats = []
    image_proc = getattr(processor, "image_processor", processor)

    with torch.inference_mode():
        for i in tqdm(range(0, len(images), batch_size), desc="Embed images", leave=False):
            batch_imgs = images[i : i + batch_size]
            inputs = image_proc(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
            f = model.get_image_features(pixel_values=pixel_values)
            feats.append(f.detach())

    feats = torch.cat(feats, dim=0)  # device == model device
    feats = F.normalize(feats.float(), dim=-1)
    return feats

def embed_texts(
    model, processor, texts: List[str], device: torch.device,
    batch_size: int = 256, is_siglip: bool = False
) -> torch.Tensor:
    """
    Compute L2-normalized text embeddings using model.get_text_features.
    Defaults preserved: batch_size=256, is_siglip=False.
    """
    model.eval()
    feats = []
    pad = "max_length" if is_siglip else True

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        # Fallback: try model.name_or_path; use CLIP tokenizer by default.
        name_or_path = getattr(model, "name_or_path", None) or "openai/clip-vit-base-patch16"
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)

    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embed texts", leave=False):
            batch_txts = texts[i : i + batch_size]
            inputs = tokenizer(text=batch_txts, truncation=True, padding=pad, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attn = inputs.get("attention_mask", None)
            attn = attn.to(device) if attn is not None else None
            f = model.get_text_features(input_ids=input_ids, attention_mask=attn)
            feats.append(f.detach())

    feats = torch.cat(feats, dim=0)
    feats = F.normalize(feats.float(), dim=-1)
    return feats
