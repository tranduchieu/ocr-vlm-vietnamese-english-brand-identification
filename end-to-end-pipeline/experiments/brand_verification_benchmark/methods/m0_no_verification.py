"""M0 -- No Brand Verification.

Per correction 1.2/1.6/section 9: M0 means no brand-EVIDENCE gate, not "skip all
finalization". accepted_brands = candidate_brands always; rejected_brands = [];
released_brands is still computed via the shared final summary-consistency audit
(contains_exact), and the shared finalize() is still applied -- so M0 differs from
M1/M2/M3 ONLY in the brand-verification decision, nothing else.

RELEASED MUST BE RE-AUDITED AFTER finalize() (fix, 2026-09-17 part 2): `finalize()` can
replace the summary entirely -- either with `summary_tu_ocr()`'s OCR-echo fallback (when
the working summary is empty but text was detected) or with the empty string "" (when
neither text nor a brand/product is present, or an absence-statement empties it). A
"released_brands" list computed only against the PRE-finalize summary would then claim
brands are visible to the user that are no longer actually present in the text they
see -- exactly what Brand F1/UBR must not misreport, since those metrics are meant to
reflect the user-facing output. Flow is now:

    decision -> cleanup (n/a for M0) -> PRELIMINARY released -> finalize() -> FINAL released
"""
from __future__ import annotations

from .common import MethodResult, brand_released, run_shared_finalize

METHOD_NAME = "m0_no_verification"


def run(image_id: str, candidate_brands: list[str], candidate_products: list[str],
        draft_summary: str, external_ocr: str, vlm_ocr: str, n_ocr_lines: int) -> MethodResult:
    r = MethodResult(
        image_id=image_id, method=METHOD_NAME,
        candidate_brands=list(candidate_brands), candidate_products=list(candidate_products),
        draft_summary=draft_summary,
    )
    # no brand-evidence gate: accept everything
    r.accepted_brands = list(candidate_brands)
    r.rejected_brands = []
    for b in candidate_brands:
        r.support_state[b] = "N/A (M0 has no verification decision)"

    # cleanup: n/a for M0, nothing was cut -- working summary is just the draft
    cleaned_summary = draft_summary

    # PRELIMINARY released: audited against the pre-finalize (but post-cleanup) summary,
    # needed only to build the brand/product string finalize() itself takes as input
    preliminary_brands = [b for b in r.accepted_brands if brand_released(cleaned_summary, b)]
    preliminary_products = [p for p in candidate_products if brand_released(cleaned_summary, p)]

    r.final_summary = run_shared_finalize(
        brand="; ".join(preliminary_brands), product="; ".join(preliminary_products),
        summary=cleaned_summary, ocr_text=external_ocr, n_ocr_lines=n_ocr_lines, vlm_ocr=vlm_ocr,
    )

    # FINAL released: re-audited against the ACTUAL final_summary -- finalize() may have
    # replaced it (summary_tu_ocr() fallback, or "" per the two empty-string rules)
    r.released_brands = [b for b in preliminary_brands if brand_released(r.final_summary, b)]
    r.released_products = [p for p in preliminary_products if brand_released(r.final_summary, p)]

    r.trace = {"note": "M0: no brand-evidence gate applied; released re-audited against "
                        "final_summary (post-finalize), not just the draft summary."}
    return r
