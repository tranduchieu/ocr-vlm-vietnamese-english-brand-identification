#!/usr/bin/env python3
"""Dataset / benchmark statistics for paper tables (reviewed GT)."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def split_multi(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dev_labels = root / "paper_dataset_release/brand_dataset/dev/labels.csv"
    bench_gt = root / "paper_dataset_release/brand_dataset/benchmark/benchmark_gt_scoring.csv"

    dev_rows = list(csv.DictReader(dev_labels.open(encoding="utf-8")))
    split_counts = Counter(r.get("split", "") for r in dev_rows)
    n_brand_imgs = sum(1 for r in dev_rows if split_multi(r.get("brands", "")))
    n_prod_imgs = sum(1 for r in dev_rows if split_multi(r.get("products", "")))

    bench = list(csv.DictReader(bench_gt.open(encoding="utf-8")))
    n_pos = sum(1 for r in bench if split_multi(r.get("gt_brands", "")))
    n_neg = len(bench) - n_pos
    n_multibrand = sum(1 for r in bench if len(split_multi(r.get("gt_brands", ""))) > 1)
    n_brand_ent = sum(len(split_multi(r.get("gt_brands", ""))) for r in bench)
    n_prod_ent = sum(len(split_multi(r.get("gt_products", ""))) for r in bench)

    ev = Counter()
    for r in bench:
        for b, e in zip(
            split_multi(r.get("gt_brands", "")),
            split_multi(r.get("gt_brand_evidence", "")) or [""] * len(split_multi(r.get("gt_brands", ""))),
        ):
            _ = b
            ev[e or "unknown"] += 1

    stats = {
        "dev_total": len(dev_rows),
        "dev_split": dict(split_counts),
        "dev_images_with_brand": n_brand_imgs,
        "dev_images_with_product": n_prod_imgs,
        "bench_images": len(bench),
        "bench_brand_positive": n_pos,
        "bench_brand_negative": n_neg,
        "bench_brand_entities": n_brand_ent,
        "bench_product_entities": n_prod_ent,
        "bench_multibrand_images": n_multibrand,
        "bench_brand_evidence_entities": dict(ev),
    }
    out = root / "end-to-end-pipeline/outputs/paper_rerun_20261113/dataset_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
