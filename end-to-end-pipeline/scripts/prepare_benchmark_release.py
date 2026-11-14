#!/usr/bin/env python3
"""GT + file-list for 582 benchmark images (reviewed labels, paper_dataset_release)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT.parent / "paper_dataset_release" / "brand_dataset" / "benchmark" / "labels.csv"
IMG = ROOT.parent / "paper_dataset_release" / "brand_dataset" / "benchmark" / "images"


def split_multi(s: str) -> str:
    parts = [x.strip() for x in (s or "").replace("|", ";").split(";") if x.strip()]
    return ";".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not LABELS.is_file():
        print(f"Missing {LABELS}", file=sys.stderr)
        return 1

    gt_path = args.out_dir / "benchmark_gt_reviewed.csv"
    list_path = args.out_dir / "benchmark_paths.txt"
    rows = list(csv.DictReader(LABELS.open(encoding="utf-8")))
    with gt_path.open("w", encoding="utf-8", newline="") as gfh, list_path.open(
        "w", encoding="utf-8"
    ) as lfh:
        wr = csv.DictWriter(
            gfh,
            fieldnames=["image_id", "filename", "gt_brands", "gt_products", "gt_brand_evidence"],
        )
        wr.writeheader()
        for r in rows:
            img = IMG / r["filename"]
            if not img.is_file():
                print(f"WARN missing {img}", file=sys.stderr)
            lfh.write(str(img.resolve()) + "\n")
            wr.writerow(
                {
                    "image_id": r["image_id"],
                    "filename": r["filename"],
                    "gt_brands": split_multi(r.get("brands") or ""),
                    "gt_products": split_multi(r.get("products") or ""),
                    "gt_brand_evidence": split_multi(r.get("brand_evidence") or ""),
                }
            )
    print(f"  gt   -> {gt_path} ({len(rows)} rows)")
    print(f"  list -> {list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
