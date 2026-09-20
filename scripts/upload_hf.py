#!/usr/bin/env python3
"""Upload a fine-tuned XERON checkpoint to Hugging Face Hub.

Usage:
    HF_TOKEN=hf_... python scripts/upload_hf.py \
        --model-dir ./output/xeron \
        --repo-id PIXELZX/XERON
"""
import argparse
import json
import os

from huggingface_hub import HfApi, upload_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo-id", required=True, help="e.g. PIXELZX/XERON")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=None, help="HF_TOKEN (or set env)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required (--token or env)")

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        print(f"Repo {args.repo_id} ready")
    except Exception as e:
        print(f"create_repo note: {e}")

    print(f"Uploading {args.model_dir} -> {args.repo_id} ...")
    upload_folder(
        repo_id=args.repo_id,
        folder_path=args.model_dir,
        token=token,
        commit_message="XERON fine-tune upload",
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()