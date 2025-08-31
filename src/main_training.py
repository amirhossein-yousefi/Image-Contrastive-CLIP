import argparse, os, random
import torch

from transformers import (
    AutoProcessor,
    AutoModel,
    TrainingArguments,
)

from src.data_utils import load_image_text_dataset
from src.collators import CollatorForCLIP
from src.trainer_custom import CLIPTrainer
from src.callbacks import RetrievalEvalCallback
from src.eval_utils import eval_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16",
                        help="CLIP or SigLIP checkpoint (e.g., openai/clip-vit-base-patch16 or google/siglip-base-patch16-224)")
    parser.add_argument("--dataset", type=str, default="flickr30k", choices=["flickr8k", "flickr30k"])
    parser.add_argument("--output_dir", type=str, default="runs/clip-finetune-flickr30k_sec")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--train_bs", type=int, default=64, help="per-device micro-batch size")
    parser.add_argument("--eval_bs", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=4, help="effective batch = train_bs * grad_accum")
    parser.add_argument("--image_resize", type=int, default=None, help="Optional override for processor.size (e.g., 224, 256)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision fp16")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # good on Ampere
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & processor (works for CLIP or SigLIP)
    attn_impl = "sdpa"
    model = AutoModel.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.float16 if args.fp16 else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)

    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.image_resize is not None and hasattr(processor, "image_processor"):
        processor.image_processor.size = {"height": args.image_resize, "width": args.image_resize}
        if hasattr(processor.image_processor, "crop_size"):
            processor.image_processor.crop_size = {"height": args.image_resize, "width": args.image_resize}

    # Data
    ds = load_image_text_dataset(args.dataset)
    ds["train"] = ds["train"].shuffle(seed=args.seed)

    # Collator: SigLIP needs padding="max_length"
    is_siglip = "siglip" in args.model_name.lower()
    collator = CollatorForCLIP(processor=processor, pad_to_max_for_siglip=is_siglip)


    model.to(device).eval()
    if "validation" in ds:
        _ = eval_split(
            "validation", ds["validation"], model, processor, device,
            eval_bs_images=args.eval_bs, eval_bs_texts=max(128, args.eval_bs), is_siglip=is_siglip
        )
    if "test" in ds:
        _ = eval_split(
            "test", ds["test"], model, processor, device,
            eval_bs_images=args.eval_bs, eval_bs_texts=max(128, args.eval_bs), is_siglip=is_siglip
        )

    # Training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        dataloader_num_workers=4,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        logging_steps=50,
        eval_strategy=("epoch" if "validation" in ds else "no"),
        save_strategy="epoch",
        save_total_limit=2,
        fp16=args.fp16,
        bf16=False,  # 30-series GPUs typically don't do native bf16
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "logs"),
        remove_unused_columns=False,
    )

    trainer = CLIPTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
    )

    # Register callback if we have a validation split
    if "validation" in ds:
        trainer.add_callback(RetrievalEvalCallback(
            processor=processor,
            split_ds=ds["validation"],
            eval_bs=args.eval_bs,
            is_siglip=is_siglip,
            every_n_epochs=1,  # change to 2/3 if you want less frequent eval
        ))

    # Train
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Post-train evaluation
    if "validation" in ds:
        _ = eval_split(
            "validation", ds["validation"], model, processor, device,
            eval_bs_images=args.eval_bs, eval_bs_texts=max(128, args.eval_bs), is_siglip=is_siglip
        )
    if "test" in ds:
        _ = eval_split(
            "test", ds["test"], model, processor, device,
            eval_bs_images=args.eval_bs, eval_bs_texts=max(128, args.eval_bs), is_siglip=is_siglip
        )

    print("Done. Artifacts saved to:", args.output_dir)

if __name__ == "__main__":
    main()
