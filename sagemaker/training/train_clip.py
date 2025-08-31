"""
SageMaker PyTorch script-mode training for CLIP/OpenCLIP.

Key features:
- Supports JSONL manifests (image path + text) with optional image root.
- Mixed precision (AMP) and gradient accumulation.
- Single-node multi-GPU or multi-node DDP (works with SageMaker launchers).
- Saves final artifacts to SM_MODEL_DIR and intermediate logs/ckpts to SM_OUTPUT_DIR/opt/ml/checkpoints.
- Emits key=value logs for CloudWatch + metric_definitions.

Relies on SageMaker env vars for data/model IO. See AWS docs.  # SM_MODEL_DIR, SM_OUTPUT_DIR, SM_CHANNEL_*  # noqa
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image

# OpenCLIP
import open_clip  # pip: open-clip-torch

# ---------- Utils ----------

def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, default)
    return v

def setup_logging(rank: int = 0) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def init_distributed_if_needed() -> Tuple[bool, int, torch.device]:
    # SageMaker sets WORLD_SIZE/RANK/LOCAL_RANK for torchrun or provides SM_NUM_GPUS, etc.  # noqa
    # See SDK docs for distribution launchers.  # noqa
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SM_NUM_GPUS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SM_LOCAL_RANK", "0")))
    is_distributed = world_size > 1 and torch.cuda.is_available()

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=torch.timedelta(seconds=1800))
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return is_distributed, local_rank, device

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

# ---------- Data ----------

@dataclass
class Sample:
    img_path: Path
    text: str

class JsonlImageTextDataset(Dataset):
    """Loads JSONL lines with at least {'image': <path>, 'text': <caption>}."""
    def __init__(self, manifest: Path, image_root: Path | None, preprocess, tokenizer, max_txt_len: int = 77):
        self.samples: List[Sample] = []
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.image_root = image_root

        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                img = obj.get("image") or obj.get("image_path") or obj.get("filepath") or obj.get("path")
                txt = obj.get("text") or obj.get("caption") or obj.get("prompt")
                if not img or not txt:
                    continue
                img_path = Path(img)
                if not img_path.is_absolute() and self.image_root:
                    img_path = self.image_root / img_path
                self.samples.append(Sample(img_path=img_path, text=str(txt)))
        if not self.samples:
            raise ValueError(f"No valid samples in {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = self.preprocess(Image.open(s.img_path).convert("RGB"))
        tokens = self.tokenizer([s.text], context_length=self.max_txt_len)[0]
        return image, tokens

def collate_batch(batch):
    images, tokens = zip(*batch)
    images = torch.stack(images, dim=0)
    tokens = torch.stack(tokens, dim=0)
    return images, tokens

# ---------- Train Loop ----------

def contrastive_loss(image_features, text_features, logit_scale: torch.Tensor):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    target = torch.arange(image_features.size(0), device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, target)
    loss_t = F.cross_entropy(logits_per_text, target)
    return (loss_i + loss_t) / 2

def save_checkpoint(model, optimizer, epoch: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "time": time.time(),
    }
    torch.save(ckpt, out_dir / f"clip_ckpt_e{epoch}.pt")

def train(args):
    is_distributed, local_rank, device = init_distributed_if_needed()
    setup_logging(rank=(dist.get_rank() if dist.is_initialized() else 0))

    # Hyperparams
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Create model + transforms
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained if args.pretrained else None,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    # Datasets
    train_manifest = Path(args.train_manifest or (Path(_env("SM_CHANNEL_TRAIN", ".")) / "manifest.jsonl"))
    val_manifest = Path(args.val_manifest) if args.val_manifest else None
    img_root = Path(args.image_root) if args.image_root else None

    ds_train = JsonlImageTextDataset(train_manifest, img_root, preprocess, tokenizer, args.max_txt_len)

    if is_distributed:
        sampler = DistributedSampler(ds_train, drop_last=True)
    else:
        sampler = None

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch,
    )

    # Optimizer
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # IO paths per SageMaker contract
    model_dir = Path(_env("SM_MODEL_DIR", "./model"))
    output_dir = Path(_env("SM_OUTPUT_DIR", "./output"))
    ckpt_dir = Path("/opt/ml/checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for step, (images, tokens) in enumerate(dl_train, start=1):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                image_features = model.module.encode_image(images) if isinstance(model, DDP) else model.encode_image(images)
                text_features = model.module.encode_text(tokens) if isinstance(model, DDP) else model.encode_text(tokens)
                logit_scale = (model.module.logit_scale if isinstance(model, DDP) else model.logit_scale).exp()
                loss = contrastive_loss(image_features, text_features, logit_scale)

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            global_step += 1

            # Emit train metrics in key=value for SageMaker metric_definitions
            if (step % args.log_every) == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                avg = epoch_loss / step
                print(f"train_loss={avg:.6f} step={global_step} epoch={epoch}", flush=True)

        if not dist.is_initialized() or dist.get_rank() == 0:
            # Save checkpoint each epoch (resumable training)
            save_checkpoint(model.module if isinstance(model, DDP) else model, optimizer, epoch, ckpt_dir)

    # Save final model to SM_MODEL_DIR (this becomes model.tar.gz artifact)
    if not dist.is_initialized() or dist.get_rank() == 0:
        final_path = model_dir / "open_clip_model.pt"
        torch.save((model.module if isinstance(model, DDP) else model).state_dict(), final_path)
        # Persist minimal config for inference
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": args.model_name,
                    "pretrained": args.pretrained,
                    "max_txt_len": args.max_txt_len,
                },
                f,
            )
        print(f"saved_model_path={final_path} epochs={args.epochs}", flush=True)

    barrier()

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train-manifest", type=str, default=None, dest="train_manifest")
    p.add_argument("--val-manifest", type=str, default=None, dest="val_manifest")
    p.add_argument("--image-root", type=str, default=None)
    # Model & training
    p.add_argument("--model-name", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--max-txt-len", type=int, default=77)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.2)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
