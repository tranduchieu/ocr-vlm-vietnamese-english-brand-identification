"""M3 -- Ours: Source-aware Verification.

Genuinely new logic -- no production equivalent exists. For every candidate brand,
support is computed SEPARATELY against external_ocr and vlm_ocr (never concatenated
before matching, unlike M2/anchor_gate()), classified into one of exactly four states
(BOTH / EXTERNAL_ONLY / SELF_ONLY / UNSUPPORTED -- CONFLICT deliberately excluded, see
configs/verifier_config.json), then a policy decides accept/reject per state.

The policy is READ FROM configs/verifier_config.json at call time -- this file does not
hardcode SELF_ONLY treatment, so it is impossible to change verification policy without
also changing (and re-hashing) that config file. Until `verifier_config.json ->
m3_source_aware.policy_frozen` is true, this module refuses to run against anything
other than the LoRA validation split explicitly (see `require_frozen_policy` below) --
this is the mechanism requested by "the benchmark code must make it impossible to
accidentally change policy during blind-test scoring without changing the config/hash."
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import (
    MethodResult, apply_brand_cut, brand_released, classify_support_state,
    run_shared_finalize, self_ocr_exact_match,
)

METHOD_NAME = "m3_source_aware"

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "verifier_config.json"

# Default policy mirrors the plan's own conservative starting point (section 6, M3
# table): accept BOTH, accept EXTERNAL_ONLY under the same rule as M1, reject
# UNSUPPORTED always. SELF_ONLY defaults to REJECT (the strictest, most conservative
# option) until a validation-set-selected rule is frozen -- this default must never be
# silently swapped for a laxer rule.
_DEFAULT_POLICY = {
    "BOTH": "accept",
    "EXTERNAL_ONLY": "accept",
    "SELF_ONLY": "reject",
    "UNSUPPORTED": "reject",
}


def load_policy() -> tuple[dict[str, str], bool]:
    """Returns (policy_dict, is_frozen). Reads live from verifier_config.json every
    call -- policy is never cached in code so a config edit always takes effect and is
    always visible to anyone hashing the config for the freeze manifest.

    SELF_ONLY additionally accepts "exact" (2026-09-17 protocol amendment, Policy B)
    -- BOTH/EXTERNAL_ONLY/UNSUPPORTED only ever parse "accept"/"reject" since they are
    not free parameters (see verifier_config.json's own notes)."""
    if not _CONFIG_PATH.exists():
        return dict(_DEFAULT_POLICY), False
    cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    m3 = cfg.get("m3_source_aware", {})
    frozen = bool(m3.get("policy_frozen", False))
    raw_policy = m3.get("policy", {})
    policy = dict(_DEFAULT_POLICY)
    for state in ("BOTH", "EXTERNAL_ONLY", "UNSUPPORTED"):
        v = raw_policy.get(state, "")
        if isinstance(v, str) and v.lower() in ("accept", "reject"):
            policy[state] = v.lower()
    v = raw_policy.get("SELF_ONLY", "")
    if isinstance(v, str) and v.lower() in ("accept", "reject", "exact"):
        policy["SELF_ONLY"] = v.lower()
        # a "TO_BE_SELECTED (...)" placeholder string falls through to _DEFAULT_POLICY
    return policy, frozen


class UnfrozenPolicyError(RuntimeError):
    pass


def run(image_id: str, candidate_brands: list[str], candidate_products: list[str],
        draft_summary: str, external_ocr: str, vlm_ocr: str, n_ocr_lines: int,
        *, allow_unfrozen_policy: bool = True) -> MethodResult:
    """`allow_unfrozen_policy` defaults True here (unit tests / validation-split runs
    need to work before freezing exists at all) -- the blind-run entrypoint in
    scripts/run_verifiers.py is responsible for passing allow_unfrozen_policy=False
    when scoring hackathon_benchmark_dataset/benchmark_gt_final.csv, which is the point
    where policy must already be frozen.
    """
    policy, frozen = load_policy()
    if not frozen and not allow_unfrozen_policy:
        raise UnfrozenPolicyError(
            "M3 policy is not frozen (verifier_config.json -> m3_source_aware.policy_frozen "
            "is false). Refusing to score the blind benchmark with an unfrozen policy. "
            "Select/freeze SELF_ONLY treatment on data/processed/val.jsonl first."
        )

    r = MethodResult(
        image_id=image_id, method=METHOD_NAME,
        candidate_brands=list(candidate_brands), candidate_products=list(candidate_products),
        draft_summary=draft_summary,
    )

    for b in candidate_brands:
        state = classify_support_state(external_ocr or "", vlm_ocr or "", b)
        r.support_state[b] = state
        r.external_supported[b] = state in ("BOTH", "EXTERNAL_ONLY")
        r.self_supported[b] = state in ("BOTH", "SELF_ONLY")

    def _decide(b: str) -> bool:
        state = r.support_state[b]
        rule = policy.get(state)
        if state == "SELF_ONLY" and rule == "exact":
            # Policy B (2026-09-17 amendment): SELF_ONLY is already a fuzzy
            # pipeline.contains() match against vlm_ocr only (classify_support_state);
            # additionally require an EXACT/verbatim match in vlm_ocr. External OCR is
            # never consulted for this state.
            return self_ocr_exact_match(vlm_ocr or "", b)
        return rule == "accept"

    r.accepted_brands = [b for b in candidate_brands if _decide(b)]
    r.rejected_brands = [b for b in candidate_brands if b not in r.accepted_brands]

    cleaned_products, cleaned_summary = apply_brand_cut(
        "; ".join(candidate_products), draft_summary, r.rejected_brands,
    )
    cleaned_products_list = [p.strip() for p in cleaned_products.split(";") if p.strip()]

    # PRELIMINARY released: audited against the pre-finalize (post-cleanup) summary
    preliminary_brands = [b for b in r.accepted_brands if brand_released(cleaned_summary, b)]
    preliminary_products = [p for p in cleaned_products_list if brand_released(cleaned_summary, p)]

    r.final_summary = run_shared_finalize(
        brand="; ".join(preliminary_brands), product="; ".join(preliminary_products),
        summary=cleaned_summary, ocr_text=external_ocr, n_ocr_lines=n_ocr_lines, vlm_ocr=vlm_ocr,
    )

    # FINAL released: re-audited against the ACTUAL final_summary -- finalize() may have
    # replaced it entirely (summary_tu_ocr() fallback, or "" per the two empty-string
    # rules); see m0_no_verification.py's module docstring for the full rationale.
    r.released_brands = [b for b in preliminary_brands if brand_released(r.final_summary, b)]
    r.released_products = [p for p in preliminary_products if brand_released(r.final_summary, p)]

    r.trace = {
        "policy_used": policy,
        "policy_frozen": frozen,
        "note": "Decision computed ext_match/self_match SEPARATELY (classify_support_state); "
                "cleanup via local apply_brand_cut() mirroring anchor_gate()'s cleanup half; "
                "released re-audited against final_summary (post-finalize).",
    }
    return r
