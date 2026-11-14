"""M4 -- Multi-view Generate-and-Verify (this phase's proposed verifier).

Thin benchmark wrapper around src/multiview_verifier.py (single definition of the decision logic,
shared with the production pipeline). Same MethodResult contract and the same downstream steps as
M0-M3 (cleanup -> release audit -> shared finalize() -> re-audit against the FINAL summary), so the
independent scorer treats all methods identically.

Differences from M3:
  1. the candidate pool is the union over several VLM views (native resolution, beam hypotheses),
     not one greedy 512x512 hypothesis;
  2. every candidate is verified against evidence independent of the hypothesis that produced it
     (external OCR) or against agreement across hypotheses (votes) - see multiview_verifier.py;
  3. accepted brands missing from the base summary are written into it (the summary is the only
     text the judge sees).

Input: `views` = {"base": <fixed-drafts record>, "<name>": <view record>, ...} for ONE image.
Policy: configs/m4_config.json -> policy merged over DEFAULT_POLICY;
`frozen` must be true before the blind set may be scored (enforced by the GPU scripts and by
scripts/run_m4.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import MethodResult, brand_released, run_shared_finalize
from multiview_verifier import (  # noqa: E402  (src/ is on sys.path via methods/common.py)
    DEFAULT_POLICY, apply_brand_cut, insert_brands_into_summary, merge_views, merged_policy, verify,
)

METHOD_NAME = "m4_multiview"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "m4_config.json"


def load_policy() -> tuple[dict, bool]:
    if not _CONFIG_PATH.exists():
        return merged_policy(None), False
    cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return merged_policy(cfg.get("policy")), bool(cfg.get("frozen", False))


def run(image_id: str, views: dict[str, dict], *, policy: dict | None = None) -> MethodResult:
    if policy is None:
        policy, _ = load_policy()
    base = views["base"]
    use = {n: views[n] for n in policy["views"] if n in views}
    use = {"base": base, **{n: v for n, v in use.items() if n != "base"}} if "base" in policy["views"] \
        else use

    cands, selfs = merge_views(use)
    accepted, info = verify(cands, base.get("external_ocr", ""), selfs, policy)
    cand_texts = [c.text for c in cands]
    acc_texts = [c.text for c in accepted]

    r = MethodResult(image_id=image_id, method=METHOD_NAME, candidate_brands=cand_texts,
                     candidate_products=list(base.get("candidate_products", [])),
                     draft_summary=base.get("draft_summary", ""))
    for t in cand_texts:
        r.support_state[t] = info[t]["state"]
        r.external_supported[t] = info[t]["ext"]
        r.self_supported[t] = info[t]["state"] in ("BOTH", "SELF_ONLY")
    r.accepted_brands = acc_texts
    r.rejected_brands = [t for t in cand_texts if t not in acc_texts]

    cleaned_products, cleaned_summary = apply_brand_cut(
        "; ".join(r.candidate_products), r.draft_summary, r.rejected_brands)
    if policy.get("insert_missing", True):
        cleaned_summary = insert_brands_into_summary(cleaned_summary, acc_texts, policy["template"])
    prods = [p.strip() for p in cleaned_products.split(";") if p.strip()]

    pre_b = [b for b in acc_texts if brand_released(cleaned_summary, b)]
    pre_p = [p for p in prods if brand_released(cleaned_summary, p)]
    r.final_summary = run_shared_finalize(
        brand="; ".join(pre_b), product="; ".join(pre_p), summary=cleaned_summary,
        ocr_text=base.get("external_ocr", ""), n_ocr_lines=base.get("n_ocr_regions", 0),
        vlm_ocr=base.get("vlm_ocr", ""))
    r.released_brands = [b for b in pre_b if brand_released(r.final_summary, b)]
    r.released_products = [p for p in pre_p if brand_released(r.final_summary, p)]
    r.trace = {"policy_used": policy, "verification": info,
               "note": "union over views, evidence-verified, accepted brands inserted into summary; "
                       "released re-audited against the FINAL summary."}
    return r
