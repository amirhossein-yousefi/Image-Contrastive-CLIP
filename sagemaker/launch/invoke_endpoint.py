import argparse, base64, json
import boto3

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--image", required=True)  # local image path
    p.add_argument("--texts", nargs="+", required=True)
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    img_b64 = base64.b64encode(open(a.image, "rb").read()).decode("utf-8")
    payload = {"task": "rank", "image_b64": img_b64, "texts": a.texts, "top_k": 5}
    smrt = boto3.client("sagemaker-runtime")
    resp = smrt.invoke_endpoint(EndpointName=a.endpoint, ContentType="application/json", Body=json.dumps(payload))
    print(resp["Body"].read().decode("utf-8"))
