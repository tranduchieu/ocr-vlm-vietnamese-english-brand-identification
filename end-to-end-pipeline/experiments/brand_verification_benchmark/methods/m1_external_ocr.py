"""M1 -- External-OCR-only Brand Verification (correction 1.3).

Precise definition: the BRAND DECISION stage uses external OCR only. This is achieved
by calling the REAL, UNMODIFIED production `pipeline.anchor_gate()` with `vlm_ocr=""`,
since anchor_gate() builds `hay = ocr_text + "\\n" + vlm_ocr` (pipeline.py:287) -- an
empty vlm_ocr contributes nothing to that haystack, collapsing the fusion to
external-OCR-only for the decision.

anchor_gate() is called directly (not reimplemented) so its cleanup half runs
identically to production for whatever it cuts -- zero drift risk for M1's own
decisions. What it does NOT give us for free is the "accepted" (pre-verbatim-filter)
brand list, since anchor_gate() reassigns `kept` in place and only returns the final
(verbatim-filtered = released) value. So `accepted_brands` is computed separately here
using the SAME imported matcher (`pipeline.contains`) with the SAME haystack
construction anchor_gate() itself uses -- not a reimplementation of new logic, just the
one intermediate value anchor_gate() doesn't expose.

The REAL vlm_ocr is still passed to the shared finalize() call downstream, per
correction 1.3 -- M1 is external-OCR-only for BRAND VERIFICATION, not an OCR-only
pipeline end to end.

RELEASED MUST BE RE-AUDITED AFTER finalize() (fix, 2026-09-17 part 2): `anchor_gate()`'s
own verbatim-filtered `kept`/`prods` output is only PRELIMINARY -- it is audited against
the pre-finalize `cleaned_summary`. `finalize()` downstream can still replace the summary
entirely (`summary_tu_ocr()` fallback, or "" per the two empty-string rules), which would
leave a stale released list claiming brands are visible that no longer are. See
methods/m0_no_verification.py's module docstring for the full rationale; same fix here:

    decision+cleanup (anchor_gate) -> PRELIMINARY released -> finalize() -> FINAL released
"""
from __future__ import annotations

import sys
from pathlib import Path

_METHODS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_METHODS_DIR.parent.parent.parent / "src"))
from pipeline import anchor_gate as pipeline_anchor_gate  # noqa: E402

from .common import MethodResult, brand_released, brand_supported, run_shared_finalize

METHOD_NAME = "m1_external_ocr_only"


def run(image_id: str, candidate_brands: list[str], candidate_products: list[str],
        draft_summary: str, external_ocr: str, vlm_ocr: str, n_ocr_lines: int) -> MethodResult:
    r = MethodResult(
        image_id=image_id, method=METHOD_NAME,
        candidate_brands=list(candidate_brands), candidate_products=list(candidate_products),
        draft_summary=draft_summary,
    )

    # accepted_brands: same haystack construction and matcher anchor_gate() itself
    # uses for its decision step, but with vlm_ocr forced empty -- this is the
    # intermediate value anchor_gate() computes internally but does not return.
    hay = external_ocr or ""
    for b in candidate_brands:
        supported = brand_supported(hay, b)
        r.external_supported[b] = supported
        r.support_state[b] = "EXTERNAL_ONLY_DECISION" if supported else "UNSUPPORTED"
    r.accepted_brands = [b for b in candidate_brands if r.external_supported.get(b)]
    r.rejected_brands = [b for b in candidate_brands if not r.external_supported.get(b)]

    # real production call, vlm_ocr forced empty for the DECISION only
    kept_str, prod_str, cleaned_summary, cut = pipeline_anchor_gate(
        "; ".join(candidate_brands), "; ".join(candidate_products), draft_summary,
        external_ocr or "", "",
    )
    preliminary_brands = [b.strip() for b in kept_str.split(";") if b.strip()]
    preliminary_products = [p.strip() for p in prod_str.split(";") if p.strip()]

    r.final_summary = run_shared_finalize(
        brand=kept_str, product=prod_str, summary=cleaned_summary,
        ocr_text=external_ocr, n_ocr_lines=n_ocr_lines, vlm_ocr=vlm_ocr,  # REAL vlm_ocr here
    )

    # FINAL released: re-audited against the ACTUAL final_summary
    r.released_brands = [b for b in preliminary_brands if brand_released(r.final_summary, b)]
    r.released_products = [p for p in preliminary_products if brand_released(r.final_summary, p)]

    r.trace = {
        "note": "Decision used pipeline.anchor_gate(..., vlm_ocr=''); shared finalize() "
                "used the REAL vlm_ocr per correction 1.3; released re-audited against "
                "final_summary (post-finalize).",
        "cut_by_anchor_gate": cut,
    }
    return r
