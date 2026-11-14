#!/usr/bin/env python3
"""Paired bootstrap confidence intervals over IMAGES (not brands), since all methods
share the same fixed drafts -- resampling images keeps the pairing valid.

No dependency on src/pipeline.py or on numpy (pure stdlib `random`) so it has no
package-install prerequisite beyond what's already available.

Run (future, once at least two methods' predictions exist):
    python3 experiments/brand_verification_benchmark/scripts/bootstrap_ci.py \\
        --gt hackathon_benchmark/hackathon_benchmark_dataset/benchmark_gt_final.csv \\
        --predictions-a experiments/brand_verification_benchmark/artifacts/predictions/m1.jsonl \\
        --predictions-b experiments/brand_verification_benchmark/artifacts/predictions/m3.jsonl \\
        --resamples 10000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import score_entities as sc  # noqa: E402

DEFAULT_GT = SCRIPTS_DIR.parent.parent.parent.parent / "hackathon_benchmark" / \
    "hackathon_benchmark_dataset" / "benchmark_gt_final.csv"


def load_predictions(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out[d["image_id"]] = d
    return out


def metric_on_subset(gt_rows: dict[str, sc.GTRow], preds: dict[str, dict],
                      image_ids: list[str]) -> dict:
    released = {iid: preds.get(iid, {}).get("released_brands", []) for iid in image_ids}
    candidate = {iid: preds.get(iid, {}).get("candidate_brands", []) for iid in image_ids}
    accepted = {iid: preds.get(iid, {}).get("accepted_brands", []) for iid in image_ids}
    subset_gt = {iid: gt_rows[iid] for iid in image_ids}
    prf1 = sc.score_entity_prf1(subset_gt, released, "brands")
    ubr = sc.score_ubr(subset_gt, released)
    cbr = sc.score_cbr(candidate, accepted, subset_gt)
    return {"brand_f1": prf1.f1, "ubr": ubr["ubr"], "cbr": cbr["cbr"]}


def bootstrap(gt_rows: dict[str, sc.GTRow], preds_a: dict[str, dict], preds_b: dict[str, dict],
              image_ids: list[str], n_resamples: int, seed: int) -> dict:
    rng = random.Random(seed)
    n = len(image_ids)
    deltas = {"brand_f1": [], "ubr": [], "cbr": []}
    for _ in range(n_resamples):
        sample = [image_ids[rng.randrange(n)] for _ in range(n)]
        ma = metric_on_subset(gt_rows, preds_a, sample)
        mb = metric_on_subset(gt_rows, preds_b, sample)
        deltas["brand_f1"].append(mb["brand_f1"] - ma["brand_f1"])
        deltas["ubr"].append(mb["ubr"] - ma["ubr"])
        deltas["cbr"].append(mb["cbr"] - ma["cbr"])

    def ci95(values: list[float]) -> tuple[float, float, float]:
        s = sorted(values)
        lo = s[int(0.025 * len(s))]
        hi = s[int(0.975 * len(s)) - 1]
        mean = sum(s) / len(s)
        return mean, lo, hi

    out = {}
    for k, vs in deltas.items():
        mean, lo, hi = ci95(vs)
        out[k] = {"mean_delta_b_minus_a": round(mean, 4),
                   "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--predictions-a", type=Path, required=True, help="baseline, e.g. m1 or m2")
    ap.add_argument("--predictions-b", type=Path, required=True, help="comparison, e.g. m3 (Ours)")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.predictions_a.exists() or not args.predictions_b.exists():
        print("Predictions not found -- run run_verifiers.py on real fixed drafts first. "
              "Not executed in this preparation phase.")
        return 2

    import csv
    with open(args.gt, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    gt_rows = {r["image_id"]: sc.GTRow(
        image_id=r["image_id"], gt_brands=sc.split_multi(r.get("gt_brands", "")),
        gt_products=sc.split_multi(r.get("gt_products", "")),
        gt_brand_evidence=sc.split_multi(r.get("gt_brand_evidence", "")),
    ) for r in rows}
    preds_a = load_predictions(args.predictions_a)
    preds_b = load_predictions(args.predictions_b)
    image_ids = sorted(gt_rows)

    result = bootstrap(gt_rows, preds_a, preds_b, image_ids, args.resamples, args.seed)
    print(json.dumps({
        "n_images": len(image_ids), "n_resamples": args.resamples, "seed": args.seed,
        "comparison": f"{args.predictions_b.stem} - {args.predictions_a.stem}",
        "results": result,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
