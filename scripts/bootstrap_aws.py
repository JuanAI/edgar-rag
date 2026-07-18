"""Create the SQS queue and S3 bucket in LocalStack (idempotent).

Run once after `docker compose up`. Safe to re-run.
"""

from __future__ import annotations

import sys

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, "/app/src")
from edgar_rag.config import get_settings  # noqa: E402


def main() -> None:
    s = get_settings()
    endpoint = s.aws_endpoint_url or None
    kw = dict(
        region_name=s.aws_region,
        aws_access_key_id=s.aws_access_key_id,
        aws_secret_access_key=s.aws_secret_access_key,
        endpoint_url=endpoint,
    )

    sqs = boto3.client("sqs", **kw)
    s3 = boto3.client("s3", **kw)

    sqs.create_queue(QueueName=s.sqs_queue_name)
    print(f"[bootstrap] SQS queue ready: {s.sqs_queue_name}")

    try:
        s3.create_bucket(Bucket=s.s3_bucket)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
    print(f"[bootstrap] S3 bucket ready: {s.s3_bucket}")


if __name__ == "__main__":
    main()
