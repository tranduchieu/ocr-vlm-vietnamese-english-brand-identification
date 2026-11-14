#!/usr/bin/env python3
"""Entity P/R/F1 for pipeline --emit-entities JSONL vs reviewed benchmark GT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "experiments" / "brand_verification_benchmark" / "scripts"
sys.path.insert(0, str(BENCH))
from score_entities import (  # noqa: E402
    GTRow,
    load_aliases,
    load_gt,
    score_entity_prf1,
    split_multi,
)


def load_pipeline_preds(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            iid = d["image_id"]
            brands = split_multi(d.get("brand") or "")
            products = split_multi(d.get("product") or "")
            out.setdefault(iid, {})
            out[iid] = {"brands": brands, "products": products, "summary": d.get("summary") or ""}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gt_rows = load_gt(args.gt)
    aliases = load_aliases()
    preds = load_pipeline_preds(args.predictions)
    missing = sorted(set(gt_rows) - set(preds))
    if missing:
        print(f"REFUSE: missing {len(missing)} ids e.g. {missing[:5]}", file=sys.stderr)
        return 2

    released_brands = {iid: v["brands"] for iid, v in preds.items()}
    released_products = {iid: v["products"] for iid, v in preds.items()}
    brand = score_entity_prf1(gt_rows, released_brands, "brands", aliases)
    product = score_entity_prf1(gt_rows, released_products, "products", aliases)

    summary_recall = summary_style(gt_rows, preds)
    report = {
        "gt_csv": str(args.gt),
        "predictions": str(args.predictions),
        "n_images": len(gt_rows),
        "brand_entity_prf1": brand.as_dict(),
        "product_entity_prf1": product.as_dict(),
        "summary_in_summary_recall": summary_recall,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def summary_style(gt_rows: dict[str, GTRow], preds: dict) -> dict:
    """Legacy substring recall (accent-stripped) for comparison with rebuild_test."""
    import re
    import unicodedata

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFD", (s or "").replace("đ", "d").replace("Đ", "D"))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
        return re.sub(r"[^a-z0-9]+", " ", s).strip()

    hit_b = tot_b = hit_p = tot_p = 0
    for iid, row in gt_rows.items():
        s = norm(preds[iid]["summary"])
        for b in row.gt_brands:
            tot_b += 1
            if norm(b) in s:
                hit_b += 1
        for p in row.gt_products:
            tot_p += 1
            if norm(p) in s:
                hit_p += 1
    return {
        "brand_recall_in_summary": round(hit_b / tot_b, 4) if tot_b else None,
        "product_recall_in_summary": round(hit_p / tot_p, 4) if tot_p else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
