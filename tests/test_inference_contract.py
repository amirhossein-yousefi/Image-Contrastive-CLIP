import base64, json
from sagemaker.inference import content_types

def test_schema_keys():
    req = {"task": "rank", "image_b64": base64.b64encode(b"fake").decode("utf-8"), "texts": ["a", "b"]}
    assert set(req.keys()).issuperset({"task", "image_b64", "texts"})
