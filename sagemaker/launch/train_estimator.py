"""
Submit a SageMaker training job for CLIP.
"""
from __future__ import annotations
import argparse
import os
import time

import sagemaker
from sagemaker.pytorch import PyTorch  # PyTorch Estimator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s3-train", required=True, help="S3 prefix with train data (expects manifest.jsonl & images/)")
    p.add_argument("--instance-type", default="ml.p4d.24xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--job-suffix", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--model-name", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    return p.parse_args()

def main():
    args = parse_args()
    sess = sagemaker.Session()
    region = sess.boto_region_name
    role = os.environ.get("SAGEMAKER_ROLE_ARN", sagemaker.get_execution_role())

    source_dir = "sagemaker"
    entry_point = "training/train_clip.py"

    # Enable torchrun/SMDDP with the SageMaker launchers (choose one)
    distribution = {"torch_distributed": {"enabled": True}}
    # Alternatives: {"pytorchddp": {"enabled": True}} or {"smdistributed": {"dataparallel": {"enabled": True}}}
    # See AWS docs for details.  # noqa

    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        distribution=distribution,
        hyperparameters={
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "model-name": args.model_name,
            "pretrained": args.pretrained,
            "amp": True,
        },
        # Let the container pip-install our extras
        requirements_file="sagemaker/requirements.txt",
        # Pin a framework if you want; otherwise SDK resolves a DLC image for you
        framework_version="2.4",
        py_version="py311",
        enable_sagemaker_metrics=True,  # surface printed key=value as CloudWatch metrics
    )

    # metric_definitions turn printed lines into CloudWatch metrics
    estimator.metric_definitions = [
        {"Name": "train:loss", "Regex": r"train_loss=([0-9.]+)"},
    ]

    # Map channels. The script reads SM_CHANNEL_TRAIN internally.
    inputs = {"train": args.s3_train}

    job_name = f"clip-train-{args.job_suffix}"
    estimator.fit(inputs, job_name=job_name)
    print("Training complete. Model artifact:", estimator.model_data)

if __name__ == "__main__":
    main()
