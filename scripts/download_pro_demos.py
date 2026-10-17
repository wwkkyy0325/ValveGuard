#!/usr/bin/env python
"""Download selected CS2 pro demos from the HuggingFace mirror.

Usage: scripts/download_pro_demos.py <list-file> <out-dir>

Requires hf_xet for the Xet-backed dataset; the cache makes reruns resumable.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "blanchon/opencs2_dataset_demo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("list_file")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    paths = [
        line.strip()
        for line in Path(args.list_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for path in paths:
        print(f"downloading {path}", flush=True)
        got = hf_hub_download(
            REPO, path, repo_type="dataset", local_dir=args.out_dir
        )
        print(f"done {got}", flush=True)


if __name__ == "__main__":
    main()
