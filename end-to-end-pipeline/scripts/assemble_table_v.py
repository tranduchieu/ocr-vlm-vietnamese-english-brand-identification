#!/usr/bin/env python3
"""Score B0..M3 prediction JSONL files and write Table V JSON + LaTeX rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "brand_verification_benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import score_entities as se  # noqa: E402

CONFIGS = [
    ("B0", "b0.jsonl"),
    ("B1", "b1.jsonl"),
    ("L0", "l0.jsonl"),
    ("M0", "m0.jsonl"),
    ("M1", "m1.jsonl"),
    ("M2", "m2.jsonl"),
    ("M3", "m3.jsonl"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gt_rows = se.load_gt(args.gt)
    aliases = se.load_aliases()
    table = {}
    for code, fname in CONFIGS:
        path = args.pred_dir / fname
        if not path.exists():
            print(f"MISSING {path}", file=sys.stderr)
            return 2
        records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        scored = se.score_all(gt_rows, records, aliases, expected_count=len(gt_rows))
        bp = scored["brand_prf1"]
        pf = scored["product_prf1"]["f1"]
        ubi = scored["ubr"]["ubi"]
        table[code] = {
            "brand_p": bp["precision"],
            "brand_r": bp["recall"],
            "brand_f1": bp["f1"],
            "product_f1": pf,
            "ubi": ubi,
            "full": scored,
        }
        print(
            f"{code}: P={bp['precision']:.4f} R={bp['recall']:.4f} "
            f"F1={bp['f1']:.4f} ProdF1={pf:.4f} UBI={ubi:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    tex_lines = []
    for code, _ in CONFIGS:
        t = table[code]
        tex_lines.append(
            f"{code} & {t['brand_p']:.4f} & {t['brand_r']:.4f} & {t['brand_f1']:.4f} & "
            f"{t['product_f1']:.4f} & {t['ubi']:.4f} \\\\"
        )
    (args.out.with_suffix(".tex.txt")).write_text("\n".join(tex_lines), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
