#!/usr/bin/env python3
"""Run M0/M1/M2/M3 over the SAME fixed_drafts.jsonl and write one JSONL per method.

No VLM/OCR call happens here -- this script only reads a frozen drafts file and applies
verification-decision + shared cleanup/finalization logic. Safe to run any number of
times without touching the GPU or re-generating anything.

Run (future, once fixed_drafts.jsonl exists):
    python3 experiments/brand_verification_benchmark/scripts/run_verifiers.py \\
        --drafts experiments/brand_verification_benchmark/artifacts/fixed_drafts/fixed_drafts.jsonl \\
        --out-dir experiments/brand_verification_benchmark/artifacts/predictions \\
        --methods m0 m1 m2 m3

M3 refuses to run against anything unless --allow-unfrozen-m3-policy is explicitly
passed OR configs/verifier_config.json marks the policy frozen -- this is the mechanism
that makes it impossible to accidentally score the blind set with an untuned M3 policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(EXP_ROOT))

from methods import m0_no_verification as m0  # noqa: E402
from methods import m1_external_ocr as m1  # noqa: E402
from methods import m2_simple_fusion as m2  # noqa: E402
from methods import m3_source_aware as m3  # noqa: E402

METHOD_MODULES = {"m0": m0, "m1": m1, "m2": m2, "m3": m3}
DEFAULT_DRAFTS = EXP_ROOT / "artifacts" / "fixed_drafts" / "fixed_drafts.jsonl"
DEFAULT_OUT_DIR = EXP_ROOT / "artifacts" / "predictions"


def load_drafts(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def run_one_method(method_key: str, drafts: list[dict], *, allow_unfrozen_m3_policy: bool) -> list[dict]:
    mod = METHOD_MODULES[method_key]
    records = []
    for d in drafts:
        kwargs = dict(
            image_id=d["image_id"],
            candidate_brands=d.get("candidate_brands", []),
            candidate_products=d.get("candidate_products", []),
            draft_summary=d.get("draft_summary", ""),
            external_ocr=d.get("external_ocr", ""),
            vlm_ocr=d.get("vlm_ocr", ""),
            n_ocr_lines=d.get("n_ocr_regions", 0),
        )
        if method_key == "m3":
            kwargs["allow_unfrozen_policy"] = allow_unfrozen_m3_policy
        result = mod.run(**kwargs)
        records.append(result.to_json_record())
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--methods", nargs="+", default=["m0", "m1", "m2", "m3"],
                     choices=list(METHOD_MODULES))
    ap.add_argument("--allow-unfrozen-m3-policy", action="store_true",
                     help="required to run m3 before verifier_config.json marks the "
                          "policy frozen -- intended for validation-split experiments only")
    args = ap.parse_args()

    if not args.drafts.exists():
        print(f"No fixed drafts at {args.drafts}. Run dump_fixed_drafts.py first "
              f"(not executed in this preparation phase -- see PRE_RUN_READINESS.md).")
        return 2

    drafts = load_drafts(args.drafts)
    print(f"loaded {len(drafts)} fixed drafts from {args.drafts}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.methods:
        try:
            records = run_one_method(key, drafts, allow_unfrozen_m3_policy=args.allow_unfrozen_m3_policy)
        except m3.UnfrozenPolicyError as exc:
            print(f"REFUSED to run {key}: {exc}")
            continue
        out_path = args.out_dir / f"{key}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {key}: {len(records)} records -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
