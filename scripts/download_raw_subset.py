"""Download a small raw CS2CD subset for mouse-view residual experiments.

This is deliberately not the full 48.9 GB dataset: 100 cheater matches and
100 clean matches, one parquet + one JSON each.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for folder in ["with_cheater_present", "no_cheater_present"]:
        for i in range(args.n):
            for ext in ("parquet", "json"):
                files.append(f"{folder}/{i}.{ext}")

    def fetch(name: str) -> str:
        path = hf_hub_download(
            repo_id=REPO,
            filename=name,
            repo_type="dataset",
            local_dir=str(out),
        )
        return path

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, f): f for f in files}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                done += 1
                if done % 20 == 0:
                    print(f"downloaded {done}/{len(files)}", flush=True)
            except Exception as exc:
                print(f"FAILED {name}: {exc}", flush=True)
    print(f"done {done}/{len(files)}")


if __name__ == "__main__":
    main()
