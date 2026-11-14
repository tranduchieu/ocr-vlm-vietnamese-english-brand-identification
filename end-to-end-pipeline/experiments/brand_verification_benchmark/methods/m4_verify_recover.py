"""M4 -- Verify-and-Recover (this phase's proposed verifier).

Thin benchmark wrapper around src/brand_verifier.py (the single definition of the decision logic,
shared with the production pipeline). Same MethodResult contract and the same downstream steps as
M0-M3 (cleanup -> release audit -> shared finalize() -> re-audit against the FINAL summary), so the
independent scorer treats all five methods identically.

What is different from M3:
  1. VLM candidates are verified exactly as M3 does (BOTH / EXTERNAL_ONLY accept, SELF_ONLY exact,
     UNSUPPORTED reject) plus an optional native-likelihood floor (policy).
  2. OCR spans that the LoRA finds far more plausible as `"brand"` than abstaining are RECOVERED
     as extra brands and written into the summary (the only text the judge sees).

Policy source: configs/verifier_config.json -> m4_verify_recover.policy (merged over
brand_verifier.DEFAULT_POLICY); `frozen` must be true before the blind set may be scored
(enforced in scripts/run_m4.py, and in scripts/score_spans.py for the GPU step).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .common import MethodResult, brand_released, run_shared_finalize
from brand_verifier import (  # noqa: E402  (src/ is put on sys.path by methods/common.py)
    DEFAULT_POLICY, apply_brand_cut, insert_brands_into_summary, recover_from_spans,
    verify_vlm_candidates,
)

METHOD_NAME = "m4_verify_recover"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "verifier_config.json"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_policy() -> tuple[dict, bool]:
    if not _CONFIG_PATH.exists():
        return copy.deepcopy(DEFAULT_POLICY), False
    cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8")).get("m4_verify_recover", {})
    return _merge(DEFAULT_POLICY, cfg.get("policy", {})), bool(cfg.get("frozen", False))


def run(image_id: str, candidate_brands: list[str], candidate_products: list[str],
        draft_summary: str, external_ocr: str, vlm_ocr: str, n_ocr_lines: int,
        *, native: dict | None = None, policy: dict | None = None) -> MethodResult:
    if policy is None:
        policy, _ = load_policy()

    r = MethodResult(image_id=image_id, method=METHOD_NAME,
                     candidate_brands=list(candidate_brands),
                     candidate_products=list(candidate_products), draft_summary=draft_summary)

    accepted_vlm, states = verify_vlm_candidates(candidate_brands, external_ocr, vlm_ocr, native, policy)
    for b in candidate_brands:
        r.support_state[b] = states[b]
        r.external_supported[b] = states[b] in ("BOTH", "EXTERNAL_ONLY")
        r.self_supported[b] = states[b] in ("BOTH", "SELF_ONLY")
    r.rejected_brands = [b for b in candidate_brands if b not in accepted_vlm]

    recovered = recover_from_spans(native, accepted_vlm, policy)
    recovered_texts = [x["text"] for x in recovered]
    r.accepted_brands = list(accepted_vlm) + recovered_texts

    cleaned_products, cleaned_summary = apply_brand_cut(
        "; ".join(candidate_products), draft_summary, r.rejected_brands)
    cleaned_summary = insert_brands_into_summary(
        cleaned_summary, recovered_texts, policy["recover"]["template"])
    cleaned_products_list = [p.strip() for p in cleaned_products.split(";") if p.strip()]

    preliminary_brands = [b for b in r.accepted_brands if brand_released(cleaned_summary, b)]
    preliminary_products = [p for p in cleaned_products_list if brand_released(cleaned_summary, p)]
    r.final_summary = run_shared_finalize(
        brand="; ".join(preliminary_brands), product="; ".join(preliminary_products),
        summary=cleaned_summary, ocr_text=external_ocr, n_ocr_lines=n_ocr_lines, vlm_ocr=vlm_ocr)
    r.released_brands = [b for b in preliminary_brands if brand_released(r.final_summary, b)]
    r.released_products = [p for p in preliminary_products if brand_released(r.final_summary, p)]

    r.trace = {
        "policy_used": policy,
        "recovered": [{"text": x["text"], "margin": round(x["margin"], 3), "n_tok": x["n_tok"],
                       "sources": x.get("sources", [])} for x in recovered],
        "note": "VLM candidates verified as M3 (+optional native floor); OCR spans recovered by "
                "LoRA-likelihood margin; recovered brands inserted into the summary; released "
                "re-audited against the FINAL summary.",
    }
    return r
