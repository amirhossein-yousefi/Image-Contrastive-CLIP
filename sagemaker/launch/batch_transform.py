from __future__ import annotations
import argparse
import sagemaker
from sagemaker.transformer import Transformer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)  # attach an existing model
    p.add_argument("--s3-input", required=True)
    p.add_argument("--s3-output", required=True)
    p.add_argument("--instance-type", default="ml.m5.xlarge")
    return p.parse_args()

def main():
    args = parse_args()
    sess = sagemaker.Session()
    transformer = Transformer(
        model_name=args.model_name,
        instance_count=1,
        instance_type=args.instance_type,
        strategy="SingleRecord",
        accept="application/json",
        assemble_with="Line",
        output_path=args.s3_output,
    )
    transformer.transform(args.s3_input, content_type="application/json", split_type="Line", logs=True, wait=True)

if __name__ == "__main__":
    main()
