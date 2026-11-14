"""Shared primitives for M0/M1/M2/M3, imported from production `pipeline.py` wherever
possible so there is exactly one definition of every matching/cleanup rule that matters.

DOES NOT MODIFY src/pipeline.py. Everything here either:
  (a) imports a real production function/constant directly, or
  (b) is new orchestration code that calls those imported pieces, or
  (c) is a documented, minimal duplication of a piece of production logic that is not
      independently importable without a small pipeline.py refactor this phase is not
      authorized to perform (see apply_brand_cut() docstring below).

candidate / accepted / released terminology (see BENCHMARK_EXPERIMENT_PLAN corrections,
section 1.6):

    candidate_brands  -- raw VLM brand candidates before any verification (post-retry).
    accepted_brands   -- candidates that pass the method's brand-EVIDENCE decision
                         (pipeline.contains(hay, b) for M1/M2/M3; always all of
                         candidate_brands for M0).
    rejected_brands   -- candidate_brands - accepted_brands.
    released_brands   -- accepted_brands that also survive the final summary-consistency
                         audit (pipeline.contains_exact(summary, b)) -- this is what
                         actually reaches the user-facing text and what Brand P/R/F1 and
                         UBR should be scored against.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_METHODS_DIR = Path(__file__).resolve().parent
_EXP_ROOT = _METHODS_DIR.parent
_PIPELINE_SRC = _EXP_ROOT.parent.parent / "src"
if str(_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_SRC))

# ---- real production primitives, imported unmodified -----------------------------
from pipeline import (  # noqa: E402
    contains as pipeline_contains,
    contains_exact as pipeline_contains_exact,
    split_multi as pipeline_split_multi,
    _don_cau as pipeline_don_cau,
    finalize as pipeline_finalize,
    OCR_CONF_MIN,
    FUZZY_THRESHOLD,
)

SupportState = Literal["BOTH", "EXTERNAL_ONLY", "SELF_ONLY", "UNSUPPORTED"]


def brand_supported(haystack: str, brand: str) -> bool:
    """Thin, literal wrapper around pipeline.contains() -- no reimplementation."""
    return pipeline_contains(haystack, brand)


def brand_released(summary: str, brand: str) -> bool:
    """Thin, literal wrapper around pipeline.contains_exact() -- no reimplementation."""
    return pipeline_contains_exact(summary, brand)


def self_ocr_exact_match(vlm_ocr: str, brand: str) -> bool:
    """Thin, literal wrapper around pipeline.contains_exact() -- no reimplementation.

    Used only by M3's SELF_ONLY_EXACT policy (2026-09-17 protocol amendment):
    a SELF_ONLY-classified candidate (fuzzy pipeline.contains() match against vlm_ocr
    only, per classify_support_state()) is additionally required to have an
    EXACT/verbatim match in vlm_ocr before being accepted. External OCR is never
    consulted here -- this is a strictly self-source check."""
    return pipeline_contains_exact(vlm_ocr, brand)


def classify_support_state(external_ocr: str, vlm_ocr: str, brand: str) -> SupportState:
    """Compute support SEPARATELY per source -- never concatenate before matching.

    This is the one thing pipeline.anchor_gate() explicitly does NOT do (it builds
    `hay = ocr_text + "\\n" + vlm_ocr` and matches once against the concatenation,
    pipeline.py:287,290-291) -- that collapsing is exactly what makes anchor_gate() M2
    (Simple Fusion) rather than a source-aware method. M3 exists specifically to avoid it.

    CONFLICT is intentionally NOT a possible return value here -- see
    configs/verifier_config.json -> m3_source_aware.conflict_state_reason.
    """
    ext = pipeline_contains(external_ocr or "", brand)
    slf = pipeline_contains(vlm_ocr or "", brand)
    if ext and slf:
        return "BOTH"
    if ext:
        return "EXTERNAL_ONLY"
    if slf:
        return "SELF_ONLY"
    return "UNSUPPORTED"


# ---- cleanup: apply an externally-supplied cut list to (product, summary) --------

# Duplicated from pipeline.anchor_gate() (pipeline.py, inside the `if cut:` block,
# approximately lines 293-307). NOT independently importable: pipeline.py does not
# expose a standalone "given a cut list, edit the summary" function -- the decision
# (`contains(hay, b)` per brand) and the cleanup (this regex block) are one function.
#
# This phase is explicitly not authorized to refactor src/pipeline.py, so the regex
# logic is copied verbatim rather than re-derived. Drift risk is real and is mitigated
# by tests/test_cleanup_equivalence.py, which asserts this function produces IDENTICAL
# (product, summary) output to the real pipeline.anchor_gate() for M1/M2 cases where
# both compute the same cut list.
#
# Recommended follow-up (not performed here, requires separate approval): extract this
# block into pipeline._apply_cut(cut_list, product, summary) inside src/pipeline.py so
# both production and this benchmark call the exact same code. See
# BENCHMARK_FEASIBILITY_REPORT.md section 8 and configs/verifier_config.json ->
# shared_cleanup.recommended_followup.
def apply_brand_cut(product: str, summary: str, cut_brands: list[str]) -> tuple[str, str]:
    prods = pipeline_split_multi(product)
    if cut_brands:
        for b in cut_brands:
            summary = re.sub(
                r"\b(?:(?:của|cua|với|voi|từ|tu|by|of|thương hiệu|thuong hieu|"
                r"nhãn hàng|nhãn hiệu|hãng)\s+){1,3}" + re.escape(b),
                "", summary, flags=re.I)
            summary = re.sub(re.escape(b) + r"\s+(và|va|&)\s+", "", summary, flags=re.I)
            summary = re.sub(r"\s+(và|va|&)\s+" + re.escape(b), "", summary, flags=re.I)
            summary = re.sub(re.escape(b), "", summary, flags=re.I)
        summary = pipeline_don_cau(summary)

        def _nfc_lower(s: str) -> str:
            import unicodedata
            return " ".join(unicodedata.normalize("NFC", (s or "").lower()).split())

        prods = [x for x in prods if not any(_nfc_lower(b) in _nfc_lower(x) for b in cut_brands)]
    return "; ".join(prods), summary


@dataclass
class BrandDecision:
    brand: str
    state: SupportState | None  # None for M0/M1/M2, set for M3
    accepted: bool
    released: bool = False  # filled in after the verbatim-audit step


@dataclass
class MethodResult:
    image_id: str
    method: str

    candidate_brands: list[str]
    candidate_products: list[str]

    support_state: dict[str, str] = field(default_factory=dict)  # brand -> state (M3 only)
    external_supported: dict[str, bool] = field(default_factory=dict)
    self_supported: dict[str, bool] = field(default_factory=dict)

    accepted_brands: list[str] = field(default_factory=list)
    rejected_brands: list[str] = field(default_factory=list)

    released_brands: list[str] = field(default_factory=list)
    released_products: list[str] = field(default_factory=list)

    draft_summary: str = ""
    final_summary: str = ""

    trace: dict = field(default_factory=dict)

    def to_json_record(self) -> dict:
        return {
            "image_id": self.image_id,
            "method": self.method,
            "candidate_brands": self.candidate_brands,
            "candidate_products": self.candidate_products,
            "external_supported": self.external_supported,
            "self_supported": self.self_supported,
            "support_state": self.support_state,
            "accepted_brands": self.accepted_brands,
            "rejected_brands": self.rejected_brands,
            "released_brands": self.released_brands,
            "released_products": self.released_products,
            "draft_summary": self.draft_summary,
            "final_summary": self.final_summary,
            "trace": self.trace,
        }


def run_shared_finalize(brand: str, product: str, summary: str,
                         ocr_text: str, n_ocr_lines: int, vlm_ocr: str) -> str:
    """Literal call to pipeline.finalize() -- identical for every method.

    Always uses the REAL vlm_ocr and REAL n_ocr_lines, even for M1 -- per correction
    1.3, M1's "external-OCR-only" property applies only to the brand-evidence DECISION
    stage, not to this shared finalization stage.
    """
    return pipeline_finalize(brand, product, summary, ocr_text, n_ocr_lines, vlm_ocr)
