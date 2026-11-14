#!/usr/bin/env python3
"""Fill missing per-brand evidence labels (align with semicolon-separated brands)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FIXES = {
    "bench_0123": "text;text",  # NUTRICARE + COLOS on packaging text
    "bench_0500": "text",  # Mẹ Sữa promotional text
    "bench_0575": "both",  # Shopee listing: product title + logo on pack
}

CSV_TARGETS: list[tuple[Path, str, str]] = [
    (
        ROOT / "paper_dataset_release/brand_dataset/benchmark/labels.csv",
        "brands",
        "brand_evidence",
    ),
    (
        ROOT / "paper_dataset_release/brand_dataset/benchmark/benchmark_gt_scoring.csv",
        "gt_brands",
        "gt_brand_evidence",
    ),
    (
        ROOT / "end-to-end-pipeline/outputs/auto_retrain_20261113/benchmark_gt_reviewed.csv",
        "gt_brands",
        "gt_brand_evidence",
    ),
    (
        ROOT / "tanahi_scr_public/benchmark/benchmark_gt_reviewed.csv",
        "gt_brands",
        "gt_brand_evidence",
    ),
]


def split_multi(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(";") if x.strip()]


def validate_row(image_id: str, brands: str, evidence: str) -> None:
    nb = len(split_multi(brands))
    ne = len(split_multi(evidence)) if evidence else 0
    if nb != ne:
        raise ValueError(f"{image_id}: {nb} brands vs {ne} evidence tokens: {brands!r} | {evidence!r}")


def patch_file(path: Path, brand_col: str, ev_col: str) -> int:
    if not path.exists():
        return 0
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    fields = rows[0].keys() if rows else []
    n = 0
    for row in rows:
        iid = row["image_id"]
        if iid not in FIXES:
            continue
        row[ev_col] = FIXES[iid]
        validate_row(iid, row[brand_col], row[ev_col])
        n += 1
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return n


def main() -> int:
    for path, bcol, ecol in CSV_TARGETS:
        c = patch_file(path, bcol, ecol)
        if c:
            print(f"patched {c} rows in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())