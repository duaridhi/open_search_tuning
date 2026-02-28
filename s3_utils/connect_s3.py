import boto3
from botocore.config import Config
def get_storage_client(local=True):
    if local:
        return boto3.client(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            config=Config(signature_version="s3v4")
        )
    else:
        # Production S3 - uses environment variables or IAM role
        return boto3.client("s3")

