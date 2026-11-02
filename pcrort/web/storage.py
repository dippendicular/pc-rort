import boto3
from botocore.exceptions import ClientError
from .config import settings

def client():
    return boto3.client("s3", endpoint_url=settings.s3_endpoint_url, aws_access_key_id=settings.s3_access_key, aws_secret_access_key=settings.s3_secret_key)
def ensure_bucket():
    try: client().head_bucket(Bucket=settings.s3_bucket)
    except ClientError: client().create_bucket(Bucket=settings.s3_bucket)
def put_file(key, fileobj, content_type="application/octet-stream"):
    ensure_bucket()
    client().upload_fileobj(fileobj, settings.s3_bucket, key, ExtraArgs={"ContentType":content_type, "ServerSideEncryption":"AES256"})
def get_file(key, fileobj): ensure_bucket(); client().download_fileobj(settings.s3_bucket, key, fileobj)

def put_path(key, path):
    with open(path, "rb") as handle: put_file(key, handle)
