from __future__ import annotations
import argparse
import sagemaker
from sagemaker.pytorch import PyTorchModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-artifact", required=True, help="s3://.../model.tar.gz from estimator.fit()")
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--mode", choices=["realtime", "async"], default="realtime")
    return p.parse_args()

def main():
    args = parse_args()
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

    model = PyTorchModel(
        model_data=args.model_artifact,
        role=role,
        entry_point="inference/inference.py",
        source_dir="sagemaker",
        framework_version="2.4",
        py_version="py311",
    )

    if args.mode == "async":
        from sagemaker.async_inference import AsyncInferenceConfig
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=args.instance_type,
            async_inference_config=AsyncInferenceConfig(),  # results via S3
        )
    else:
        predictor = model.deploy(initial_instance_count=1, instance_type=args.instance_type)

    print("Endpoint:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
