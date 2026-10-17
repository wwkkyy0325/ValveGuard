#!/usr/bin/env python
"""Download CS2CD per-match JSON files (rank and match metadata).

Usage: scripts/download_cs2cd_json.py <list-file> <out-dir>

The list holds paths such as ``with_cheater_present/12.json``.  JSON files are
~2 MB each, so rank replication does not require the 48.9 GB parquet set.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("list_file")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    names = [x.strip() for x in Path(args.list_file).read_text().splitlines() if x.strip()]

    def fetch(name: str) -> str:
        return hf_hub_download(REPO, name, repo_type="dataset", local_dir=args.out_dir)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, n): n for n in names}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"downloaded {done}/{len(names)}", flush=True)
            except Exception as exc:
                print(f"FAILED {name}: {exc}", flush=True)
    print(f"done {done}/{len(names)}")


if __name__ == "__main__":
    main()
