"""M2 -- Simple Fusion. This IS current production verification behavior.

Calls the REAL, UNMODIFIED production `pipeline.anchor_gate()` with the real
`vlm_ocr` -- i.e. exactly what src/pipeline.py already does today for every image.
No new logic. This is deliberately the strongest possible baseline against M3: any
difference measured between M2 and M3 is a difference in FUSION POLICY (equal-trust
concatenation vs. source-aware classification), not a difference in matcher
implementation, since both use the same imported `pipeline.contains()`.

RELEASED MUST BE RE-AUDITED AFTER finalize() -- see m1_external_ocr.py / m0's module
docstring for the rationale (finalize() can replace the summary via summary_tu_ocr() or
empty it, staling out a released list computed only pre-finalize).
"""
from __future__ import annotations

import sys
from pathlib import Path

_METHODS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_METHODS_DIR.parent.parent.parent / "src"))
from pipeline import anchor_gate as pipeline_anchor_gate  # noqa: E402

from .common import MethodResult, brand_released, brand_supported, run_shared_finalize

METHOD_NAME = "m2_simple_fusion"


def run(image_id: str, candidate_brands: list[str], candidate_products: list[str],
        draft_summary: str, external_ocr: str, vlm_ocr: str, n_ocr_lines: int) -> MethodResult:
    r = MethodResult(
        image_id=image_id, method=METHOD_NAME,
        candidate_brands=list(candidate_brands), candidate_products=list(candidate_products),
        draft_summary=draft_summary,
    )

    # same intermediate-value caveat as M1: anchor_gate() doesn't expose the
    # pre-verbatim-filter "accepted" list, so it's recomputed with the exact same
    # concatenated-haystack construction anchor_gate() itself uses internally
    # (pipeline.py:287).
    hay = (external_ocr or "") + "\n" + (vlm_ocr or "")
    for b in candidate_brands:
        supported = brand_supported(hay, b)
        r.external_supported[b] = brand_supported(external_ocr or "", b)
        r.self_supported[b] = brand_supported(vlm_ocr or "", b)
        r.support_state[b] = "FUSED_SUPPORTED" if supported else "UNSUPPORTED"
    r.accepted_brands = [b for b in candidate_brands if r.support_state.get(b) == "FUSED_SUPPORTED"]
    r.rejected_brands = [b for b in candidate_brands if r.support_state.get(b) != "FUSED_SUPPORTED"]

    kept_str, prod_str, cleaned_summary, cut = pipeline_anchor_gate(
        "; ".join(candidate_brands), "; ".join(candidate_products), draft_summary,
        external_ocr or "", vlm_ocr or "",
    )
    preliminary_brands = [b.strip() for b in kept_str.split(";") if b.strip()]
    preliminary_products = [p.strip() for p in prod_str.split(";") if p.strip()]

    r.final_summary = run_shared_finalize(
        brand=kept_str, product=prod_str, summary=cleaned_summary,
        ocr_text=external_ocr, n_ocr_lines=n_ocr_lines, vlm_ocr=vlm_ocr,
    )

    # FINAL released: re-audited against the ACTUAL final_summary
    r.released_brands = [b for b in preliminary_brands if brand_released(r.final_summary, b)]
    r.released_products = [p for p in preliminary_products if brand_released(r.final_summary, p)]

    r.trace = {
        "note": "Decision + cleanup used pipeline.anchor_gate() completely unmodified "
                "with the real vlm_ocr -- this IS today's production behavior. Released "
                "re-audited against final_summary (post-finalize).",
        "cut_by_anchor_gate": cut,
    }
    return r
