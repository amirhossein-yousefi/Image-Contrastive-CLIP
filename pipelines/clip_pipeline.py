from __future__ import annotations
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.pytorch import PyTorch
import sagemaker

def build_pipeline(role: str, region: str, s3_train: str, model_package_group_name: str) -> Pipeline:
    sess = sagemaker.Session()
    est = PyTorch(
        entry_point="sagemaker/training/train_clip.py",
        source_dir="sagemaker",
        role=role,
        framework_version="2.4",
        py_version="py311",
        instance_type="ml.g5.4xlarge",
        instance_count=1,
        hyperparameters={"epochs": 2, "batch-size": 128, "model-name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
        requirements_file="sagemaker/requirements.txt",
        enable_sagemaker_metrics=True,
    )
    train_step = TrainingStep(name="TrainCLIP", estimator=est, inputs={"train": s3_train})
    register_step = RegisterModel(
        name="RegisterCLIP",
        estimator=est,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
    )
    return Pipeline(
        name="CLIPPipeline",
        steps=[train_step, register_step],
        sagemaker_session=sess,
        role=role,
    )
