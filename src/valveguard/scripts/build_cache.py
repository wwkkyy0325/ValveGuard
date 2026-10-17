"""Build a memmap cache from the CS2CD context-window tar."""
from __future__ import annotations

import argparse
from pathlib import Path

from valveguard.data.tar_index import build_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="context_windows_256.tar")
    ap.add_argument("--out", required=True, help="cache directory")
    ap.add_argument("--include-augmented", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    size = Path(args.tar).stat().st_size
    expected = 9_385_585_152
    if args.limit is None and size < expected * 0.99:
        raise SystemExit(
            f"tar is incomplete: {size} bytes, expected ~{expected}. "
            "Wait for the download or pass --limit for a smoke test."
        )
    index = build_cache(
        args.tar, out,
        include_augmented=args.include_augmented,
        limit=args.limit,
    )
    print(f"wrote {len(index)} windows to {out}")
    print(index.groupby(["label", "source"]).size().to_string())


if __name__ == "__main__":
    main()
