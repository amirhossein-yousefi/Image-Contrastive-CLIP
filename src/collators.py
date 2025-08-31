from dataclasses import dataclass
from typing import Any, Dict, List
import random
import torch
from transformers import AutoTokenizer

@dataclass
class CollatorForCLIP:
    processor: Any
    pad_to_max_for_siglip: bool

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [f["image"] for f in features]
        texts  = [random.choice(f["captions"]) for f in features]

        # Use sub-components explicitly (avoids CLIPProcessor.__call__ fast/slow pitfall)
        image_proc = getattr(self.processor, "image_processor", self.processor)
        tokenizer  = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            name_or_path = getattr(self.processor, "name_or_path", None) or "openai/clip-vit-base-patch16"
            tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)

        pixel = image_proc(images=images, return_tensors="pt")
        text  = tokenizer(
            text=texts,
            truncation=True,
            padding=("max_length" if self.pad_to_max_for_siglip else True),
            return_tensors="pt",
        )
        return {**pixel, **text}
