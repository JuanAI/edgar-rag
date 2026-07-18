"""Seed the pipeline with real, public SEC EDGAR 10-K filings.

For each requested ticker we:
  1. resolve its CIK,
  2. find its most recent 10-K via the SEC submissions API,
  3. download the primary filing document,
  4. upload it to S3 (LocalStack), and
  5. enqueue an ingestion message on SQS.

The worker picks the messages up and runs the full ingest pipeline.

SEC asks for a descriptive User-Agent with contact info — set
EDGAR_RAG_SEC_USER_AGENT (default below) so requests aren't throttled/blocked.
"""
from __future__ import annotations

import json
import os
import sys
import time

import boto3
import httpx
from botocore.exceptions import ClientError

sys.path.insert(0, "/app/src")
from edgar_rag.config import get_settings  # noqa: E402

SEC_UA = os.environ.get("EDGAR_RAG_SEC_USER_AGENT", "edgar-rag-demo contact@example.com")
HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
DEFAULT_TICKERS = ["AAPL", "MSFT", "KO"]


def _client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True)


def resolve_cik(client: httpx.Client, ticker: str) -> tuple[str, str]:
    resp = client.get("https://www.sec.gov/files/company_tickers.json")
    resp.raise_for_status()
    for row in resp.json().values():
        if row["ticker"].upper() == ticker.upper():
            return f"{int(row['cik_str']):010d}", row["title"]
    raise ValueError(f"ticker not found on EDGAR: {ticker}")


def latest_10k(client: httpx.Client, cik: str) -> tuple[str, str]:
    resp = client.get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    for form, accession, primary in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], strict=False
    ):
        if form == "10-K":
            return accession.replace("-", ""), primary
    raise ValueError(f"no 10-K found for CIK {cik}")


def download_filing(client: httpx.Client, cik: str, accession: str, primary: str) -> bytes:
    cik_int = int(cik)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{primary}"
    resp = client.get(url)
    resp.raise_for_status()
    return resp.content


def main() -> None:
    tickers = sys.argv[1:] or DEFAULT_TICKERS
    s = get_settings()
    endpoint = s.aws_endpoint_url or None
    aws = dict(
        region_name=s.aws_region,
        aws_access_key_id=s.aws_access_key_id,
        aws_secret_access_key=s.aws_secret_access_key,
        endpoint_url=endpoint,
    )
    s3 = boto3.client("s3", **aws)
    sqs = boto3.client("sqs", **aws)

    # Ensure the queue and bucket exist so seeding works standalone (both calls
    # are idempotent), rather than requiring a separate bootstrap step first.
    queue_url = sqs.create_queue(QueueName=s.sqs_queue_name)["QueueUrl"]
    try:
        s3.create_bucket(Bucket=s.s3_bucket)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise

    with _client() as client:
        for ticker in tickers:
            cik, title = resolve_cik(client, ticker)
            accession, primary = latest_10k(client, cik)
            body = download_filing(client, cik, accession, primary)
            time.sleep(0.3)  # be polite to SEC

            document_id = f"{ticker.upper()}-10K-{accession}"
            key = f"{ticker.upper()}/{accession}/{primary}"
            s3.put_object(Bucket=s.s3_bucket, Key=key, Body=body)

            message = {
                "document_id": document_id,
                "s3_bucket": s.s3_bucket,
                "s3_key": key,
                "tenant": ticker.upper(),
                "doc_type": "10-K",
                "metadata": {"company": title, "accession": accession},
            }
            sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message))
            print(f"[seed] queued {document_id} ({len(body):,} bytes) -> tenant={ticker.upper()}")

    print(f"[seed] done. Seeded {len(tickers)} filing(s). The worker is now ingesting them.")


if __name__ == "__main__":
    main()
