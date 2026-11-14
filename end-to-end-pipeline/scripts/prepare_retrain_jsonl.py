#!/usr/bin/env python3
"""Build train/val/dev_test JSONL from paper_dataset_release dev labels (post-audit)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = ROOT.parent / "paper_dataset_release" / "brand_dataset" / "dev" / "labels.csv"
DEFAULT_IMG_DIR = ROOT.parent / "paper_dataset_release" / "brand_dataset" / "dev" / "images"
OUT_DIR = ROOT / "data" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not args.labels.is_file():
        print(f"Missing {args.labels}", file=sys.stderr)
        return 1

    buckets: dict[str, list[dict]] = {"train": [], "val": [], "dev_test": []}
    with args.labels.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sp = (r.get("split") or "train").strip()
            if sp not in buckets:
                sp = "train"
            img = args.image_dir / r["filename"]
            if not img.is_file():
                print(f"WARN missing image {img}", file=sys.stderr)
            try:
                rel = img.relative_to(ROOT)
            except ValueError:
                rel = Path("..") / img.relative_to(ROOT.parent)
            rec = {
                "image": str(rel).replace("\\", "/"),
                "image_id": r["image_id"],
                "ocr_text": r.get("ocr_text") or "",
                "ocr_text_model": r.get("ocr_text_model") or "",
                "brand": r.get("brands") or "",
                "product": r.get("products") or "",
                "summary": r.get("summary") or "",
            }
            buckets[sp].append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "train": args.out_dir / "train.jsonl",
        "val": args.out_dir / "val.jsonl",
        "dev_test": args.out_dir / "dev_test.jsonl",
    }
    for sp, path in mapping.items():
        with path.open("w", encoding="utf-8") as fh:
            for rec in buckets[sp]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {sp:9s} -> {path}  ({len(buckets[sp])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
