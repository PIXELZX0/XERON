#!/usr/bin/env python3
"""Upload XERON artifacts (train items / datasets / trained model) to an S3-compatible store.

S3-compatible: AWS S3, Cloudflare R2, Backblaze B2, MinIO, ...

Credentials via env (or --flag):
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / S3_ENDPOINT / S3_REGION / S3_BUCKET

Usage:
    S3_ENDPOINT=https://<endpoint> S3_BUCKET=xeron \\
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
    python scripts/upload_s3.py --key train_items_all_v2.pt /home/yuchan/laya-models/train_items_all_v2.pt

    # 학습 결과(모델 디렉토리) 통째로 업로드
    python scripts/upload_s3.py --key models/xeron-all-v2 --dir ./output/xeron-all-v2
"""
import argparse
import os

import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="S3 object key")
    ap.add_argument("--file", default=None, help="local file to upload (or use --dir)")
    ap.add_argument("--dir", default=None, help="local directory to upload recursively")
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "xeron"))
    ap.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT", ""))
    ap.add_argument("--region", default=os.environ.get("S3_REGION", "us-east-1"))
    args = ap.parse_args()

    if not (args.file or args.dir):
        ap.error("provide --file or --dir")

    ak = os.environ["AWS_ACCESS_KEY_ID"]
    sk = os.environ["AWS_SECRET_ACCESS_KEY"]

    is_aws = "amazonaws.com" in args.endpoint
    client_kwargs = dict(aws_access_key_id=ak, aws_secret_access_key=sk)
    if not is_aws:
        client_kwargs["endpoint_url"] = args.endpoint
    client_kwargs["region_name"] = args.region
    s3 = boto3.client("s3", **client_kwargs)

    if args.file:
        s3.upload_file(args.file, args.bucket, args.key)
        print(f"uploaded {args.file} -> s3://{args.bucket}/{args.key}")
    else:
        n = 0
        for root, _dirs, files in os.walk(args.dir):
            for f in files:
                local = os.path.join(root, f)
                rel = os.path.relpath(local, args.dir)
                key = f"{args.key.rstrip('/')}/{rel}"
                s3.upload_file(local, args.bucket, key)
                n += 1
        print(f"uploaded {n} files -> s3://{args.bucket}/{args.key}/")


if __name__ == "__main__":
    main()