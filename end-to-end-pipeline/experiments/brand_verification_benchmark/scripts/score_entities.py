#!/usr/bin/env python3
"""Independent deterministic entity scorer.

MUST NOT import pipeline.contains / pipeline.contains_exact / pipeline._fuzzy_in --
enforced by test_scorer_independence.py, which fails the whole test suite if this
module (or anything it imports) pulls in `src/pipeline`.

Implements only the normalization + partial-match rule specified in
hackathon-brief.md section 5.1 and configs/evaluator_aliases.json -- no
SequenceMatcher, no embeddings, no LLM judge, no reuse of the verifier's fuzzy
thresholds (pipeline.FUZZY_THRESHOLD=0.85 is never referenced here).

Metric -> data-level mapping (must stay explicit, see
BENCHMARK_EXPERIMENT_PLAN corrections section 1.6):

    Brand/Product P/R/F1  -> released_brands / released_products  vs GT
    UBR / UBI             -> released_brands                      vs GT (FP side of the
                              SAME one-to-one matching used for precision; see
                              score_ubr()'s docstring for the UBR = 1 - Precision identity)
    CBR                   -> candidate_brands -> accepted_brands  (NOT released_brands)
    RBL                   -> rejected_brands  vs final_summary

MATCHING IS ONE-TO-ONE (fix, 2026-09-17): a single prediction can match at most one GT
entity and a single GT entity can match at most one prediction, via deterministic
maximum bipartite matching (see one_to_one_match()). The earlier version of this file
asked "which GT entities have >=1 matching prediction?" and "which predictions have >=1
matching GT?" as two SEPARATE questions -- that double-counts many-to-one and
one-to-one-many cases (e.g. GT=['Nike'], predictions=['Nike','Nike Vietnam'] would have
scored TP=2 instead of the correct TP=1, FP=1).

NEGATIVE IMAGES CONTRIBUTE FALSE POSITIVES (fix, 2026-09-17): the earlier version
skipped images with an empty GT entity list entirely ("if not gts: continue"), which
silently excluded brand-negative images from the precision denominator -- a prediction
hallucinated on a brand-negative image was invisible to Brand Precision. Negative images
are no longer skipped; they simply have zero GT nodes to match against, so every
prediction on them becomes a false positive via the same one_to_one_match() code path
(no special-casing needed).
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------- normalization ----

def normalize(s: str) -> str:
    """NFC -> casefold -> strip punctuation outside [\\w\\s] -> collapse whitespace.

    Mirrors configs/evaluator_aliases.json -> normalization. Diacritics are NOT
    stripped (GT brand strings intentionally preserve them, see that config's
    accent_handling note).
    """
    s = unicodedata.normalize("NFC", s or "")
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_aliases(aliases_path: Path | None = None) -> dict[str, list[str]]:
    if aliases_path is None:
        aliases_path = Path(__file__).resolve().parent.parent / "configs" / "evaluator_aliases.json"
    if not aliases_path.exists():
        return {}
    cfg = json.loads(aliases_path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for entry in cfg.get("aliases", []):
        out[entry["canonical"]] = entry.get("accepted_variants", [])
    return out


_VOLUME_RE = re.compile(
    r"(?i)\b\d+(?:[.,]\d+)?\s*(?:ml|mililit|millilitre|l|lit|lít|liter|litre)\b"
)


def strip_volume_tokens(text: str) -> str:
    """Drop declarative quantity tokens (e.g. 1400ml) for relaxed product matching."""
    return _VOLUME_RE.sub(" ", text or "")


def product_tokens(text: str) -> list[str]:
    return [t for t in normalize(strip_volume_tokens(text)).split() if t]


def product_entity_matches(
    gt_name: str,
    predicted_text: str,
    gt_brands: list[str] | None = None,
    aliases: dict[str, list[str]] | None = None,
) -> bool:
    """Relaxed product match: volumes optional; GT/pred may differ by brand token in product string."""
    if entity_matches(gt_name, predicted_text, aliases):
        return True
    gt_t = product_tokens(gt_name)
    pr_t = product_tokens(predicted_text)
    if not gt_t:
        return False
    gt_set, pr_set = set(gt_t), set(pr_t)
    if gt_set <= pr_set or pr_set <= gt_set:
        return True
    brand_toks = {normalize(b) for b in (gt_brands or []) if normalize(b)}
    gt_core = gt_set - brand_toks
    pr_core = pr_set - brand_toks
    if gt_core and (gt_core <= pr_set or gt_core <= pr_set | brand_toks):
        return True
    if pr_core and (pr_core <= gt_set or pr_core <= gt_set | brand_toks):
        return True
    if gt_core and pr_core and (gt_core == pr_core or gt_core <= pr_core or pr_core <= gt_core):
        return True
    return False


def entity_matches(gt_name: str, predicted_text: str, aliases: dict[str, list[str]] | None = None) -> bool:
    """Partial/containment match per hackathon-brief.md 5.1: normalized GT must appear
    as a substring of the normalized prediction. GT='vaseline' matches
    prediction='vaseline vietnam'.

    Aliases only ADD alternative surface forms of the same GT entity to check for
    containment -- they never let one match count for more than one (gt, pred) edge,
    since edge existence (True/False) is all downstream one_to_one_match() consumes;
    ties/duplicates are resolved by the matching algorithm, not by this function.
    """
    g = normalize(gt_name)
    p = normalize(predicted_text)
    if not g:
        return False
    if g in p:
        return True
    if aliases:
        for variant in aliases.get(gt_name, []):
            if normalize(variant) and normalize(variant) in p:
                return True
    return False


def any_entity_matches(gt_name: str, predicted_list: list[str],
                        aliases: dict[str, list[str]] | None = None) -> bool:
    return any(entity_matches(gt_name, pred, aliases) for pred in predicted_list)


def text_contains(haystack: str, needle: str) -> bool:
    """Deterministic containment check used for RBL -- normalized substring only, no
    fuzzy matching, independent of pipeline.contains_exact()."""
    n = normalize(needle)
    return bool(n) and n in normalize(haystack)


# --------------------------------------------------------- one-to-one entity matching

def one_to_one_match(
    gts: list[str],
    preds: list[str],
    aliases: dict[str, list[str]] | None = None,
    edge_predicate=None,
) -> list[tuple[int, int]]:
    """Deterministic MAXIMUM bipartite matching between GT entities and predictions,
    using entity_matches() as the edge predicate. Each GT index and each prediction
    index appears in at most one returned (gt_idx, pred_idx) pair.

    Algorithm: Kuhn's augmenting-path algorithm (standard, O(V*E), more than sufficient
    for the tiny per-image entity counts here -- no external graph/ML library needed).

    TIE-BREAKING (documented, per task instructions): when multiple maximum matchings
    exist, this implementation processes GT entities in their given list order, and for
    each GT tries candidate predictions in their given list order, preferring the
    EARLIEST unmatched prediction and reassigning an already-matched prediction to its
    other GT candidates (via the standard augmenting-path search) only if that GT has no
    other option. Concretely:

      GT=['Nike'], preds=['Nike', 'Nike Vietnam']
          -> matches ('Nike' idx0 <-> pred idx0 'Nike'); pred idx1 becomes FP.
      GT=['Nike', 'Nike Vietnam'], preds=['Nike Vietnam']
          -> matches (GT idx0 'Nike' <-> pred idx0); GT idx1 'Nike Vietnam' becomes FN
             (a single prediction never counts as matching two GT entities).

    This tie-break is arbitrary among equally-valid maximum matchings but is fully
    deterministic given fixed input order, which is what matters for reproducibility.
    """
    match_edge = edge_predicate or (lambda g, p: entity_matches(g, p, aliases))
    n_gt, n_pred = len(gts), len(preds)
    adjacency = [[j for j in range(n_pred) if match_edge(gts[i], preds[j])] for i in range(n_gt)]
    match_for_pred: list[int] = [-1] * n_pred  # pred_idx -> matched gt_idx, or -1

    def try_augment(gt_idx: int, visited: set[int]) -> bool:
        for pred_idx in adjacency[gt_idx]:
            if pred_idx in visited:
                continue
            visited.add(pred_idx)
            if match_for_pred[pred_idx] == -1 or try_augment(match_for_pred[pred_idx], visited):
                match_for_pred[pred_idx] = gt_idx
                return True
        return False

    for gt_idx in range(n_gt):
        try_augment(gt_idx, set())

    return sorted((gt_idx, pred_idx) for pred_idx, gt_idx in enumerate(match_for_pred) if gt_idx != -1)


# ---------------------------------------------------------------------- GT loading --

def split_multi(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(";") if x.strip()]


@dataclass
class GTRow:
    image_id: str
    gt_brands: list[str]
    gt_products: list[str]
    gt_brand_evidence: list[str]


def load_gt(gt_csv: Path) -> dict[str, GTRow]:
    import csv
    out: dict[str, GTRow] = {}
    with open(gt_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["image_id"]] = GTRow(
                image_id=r["image_id"],
                gt_brands=split_multi(r.get("gt_brands", "")),
                gt_products=split_multi(r.get("gt_products", "")),
                gt_brand_evidence=split_multi(r.get("gt_brand_evidence", "")),
            )
    return out


# --------------------------------------------------------------------- aggregation --

@dataclass
class PRF1:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4)}


def score_entity_prf1(gt_rows: dict[str, GTRow], predictions: dict[str, list[str]],
                       entity_field: str, aliases: dict[str, list[str]] | None = None) -> PRF1:
    """Micro P/R/F1 across ALL images, INCLUDING GT-negative ones (fix, 2026-09-17: a
    prediction on a negative image is a false positive, not a skipped/ignored case).

    `entity_field` is 'brands' or 'products'. Uses one_to_one_match() per image: TP =
    number of matched pairs, FP = predictions - TP, FN = GT entities - TP.
    """
    agg = PRF1()
    for image_id, row in gt_rows.items():
        gts = row.gt_brands if entity_field == "brands" else row.gt_products
        preds = predictions.get(image_id, [])
        if entity_field == "products":
            brands = row.gt_brands
            pairs = one_to_one_match(
                gts,
                preds,
                aliases,
                edge_predicate=lambda g, p: product_entity_matches(g, p, brands, aliases),
            )
        else:
            pairs = one_to_one_match(gts, preds, aliases)
        tp = len(pairs)
        agg.tp += tp
        agg.fp += len(preds) - tp
        agg.fn += len(gts) - tp
    return agg


def score_ubr(gt_rows: dict[str, GTRow], released_brands: dict[str, list[str]],
              aliases: dict[str, list[str]] | None = None) -> dict:
    """UBR = unsupported released brands / all released brands, using the SAME
    one-to-one matching as Brand Precision (fix, 2026-09-17). Under that shared
    definition, unsupported released brands == FP_brand, so:

        UBR = FP / (TP + FP) = 1 - Precision   (when TP + FP > 0)

    This identity is asserted by tests/test_scorer_metrics.py::test_ubr_equals_one_minus_precision.
    UBI (image-level) = fraction of images with >=1 unsupported (FP) released brand,
    unaffected by this fix's matching change other than now agreeing with precision.
    """
    unsupported = 0
    total_released = 0
    images_with_unsupported = 0
    for image_id, row in gt_rows.items():
        preds = released_brands.get(image_id, [])
        pairs = one_to_one_match(row.gt_brands, preds, aliases)
        tp = len(pairs)
        fp = len(preds) - tp
        total_released += len(preds)
        unsupported += fp
        if fp > 0:
            images_with_unsupported += 1
    ubr = unsupported / total_released if total_released else 0.0
    ubi = images_with_unsupported / len(gt_rows) if gt_rows else 0.0
    return {
        "unsupported_released": unsupported, "total_released": total_released,
        "ubr": round(ubr, 4), "ubi": round(ubi, 4),
        "images_with_unsupported": images_with_unsupported, "total_images": len(gt_rows),
    }


def score_cbr(candidate_brands: dict[str, list[str]], accepted_brands: dict[str, list[str]],
              gt_rows: dict[str, GTRow], aliases: dict[str, list[str]] | None = None) -> dict:
    """CBR = correct draft brand CANDIDATES retained in ACCEPTED brands.

    'correct candidate' is determined by one_to_one_match(GT, candidate_brands) (fix,
    2026-09-17: previously used "any GT matches this candidate", which could let two
    near-duplicate candidates both count as separately 'correct' against the SAME single
    GT entity, inflating the denominator). Each correct candidate is identified by its
    INDEX in candidate_brands (via the matching), so duplicate-string candidates are not
    conflated.

    Retention is checked by STRING VALUE against accepted_brands as a MULTISET
    (collections.Counter): since accepted_brands is always a subset of candidate_brands
    (the same strings, filtered) for every method in this benchmark, this correctly
    handles the common case (no duplicate identical candidate strings) and the
    duplicate-string edge case without needing per-index provenance plumbed through
    MethodResult. Denominator is always ONLY correct raw candidates -- never all GT
    brands for the image (see tests/test_methods.py::test_cbr_denominator_is_raw_candidates_not_all_gt_brands).
    """
    correct_candidates_total = 0
    correct_retained_total = 0
    for image_id, row in gt_rows.items():
        cands = candidate_brands.get(image_id, [])
        accepted_counter = Counter(accepted_brands.get(image_id, []))
        pairs = one_to_one_match(row.gt_brands, cands, aliases)
        correct_indices = sorted(pred_idx for _, pred_idx in pairs)
        correct_candidates_total += len(correct_indices)

        used = Counter()
        for idx in correct_indices:
            s = cands[idx]
            if used[s] < accepted_counter[s]:
                used[s] += 1
                correct_retained_total += 1
    cbr = (correct_retained_total / correct_candidates_total) if correct_candidates_total else None
    return {
        "correct_candidates": correct_candidates_total,
        "correct_retained": correct_retained_total,
        "cbr": round(cbr, 4) if cbr is not None else None,
    }


def score_rbl(rejected_brands: dict[str, list[str]], final_summaries: dict[str, str]) -> dict:
    total_rejected = 0
    leaked = 0
    for image_id, rejected in rejected_brands.items():
        summary = final_summaries.get(image_id, "")
        for b in rejected:
            total_rejected += 1
            if text_contains(summary, b):
                leaked += 1
    rbl = leaked / total_rejected if total_rejected else 0.0
    return {"leaked": leaked, "total_rejected": total_rejected, "rbl": round(rbl, 4)}


def score_false_brand_release_rate(gt_rows: dict[str, GTRow],
                                    released_brands: dict[str, list[str]]) -> dict:
    """Plan section 14.1 -- brand-negative images that release >=1 brand anyway."""
    negatives = [iid for iid, r in gt_rows.items() if not r.gt_brands]
    false_releases = sum(1 for iid in negatives if released_brands.get(iid))
    rate = false_releases / len(negatives) if negatives else 0.0
    return {"brand_negative_images": len(negatives), "false_releases": false_releases,
            "false_brand_release_rate": round(rate, 4)}


def _predictions_map(records: list[dict], field_name: str) -> dict[str, list[str]]:
    return {d["image_id"]: d.get(field_name, []) for d in records}


class IncompletePredictionsError(RuntimeError):
    """Raised by assert_complete_predictions() -- scoring an incomplete prediction set
    silently would produce a Brand F1/UBR/CBR that looks legitimate but is computed over
    the wrong denominator. This is a harder, more important guard than additional unit
    tests: it fires on the REAL data path every time score_entities.py is run, not just
    in a test suite."""


def assert_complete_predictions(gt_rows: dict[str, GTRow], records: list[dict],
                                 expected_count: int | None = None) -> None:
    """Predictions must cover EXACTLY the GT image_ids -- no duplicates, nothing
    missing, nothing extra -- before any metric is computed from them.

    `expected_count` defaults to len(gt_rows) (so this generalizes to scoring against
    any GT file, e.g. a 427-row validation GT, not just the 582-row blind benchmark) but
    can be pinned explicitly as an extra sanity check.
    """
    pred_ids = [r["image_id"] for r in records]
    dup = sorted({i for i in pred_ids if pred_ids.count(i) > 1})
    missing = sorted(set(gt_rows) - set(pred_ids))
    extra = sorted(set(pred_ids) - set(gt_rows))
    n_expected = expected_count if expected_count is not None else len(gt_rows)

    errors = []
    if dup:
        errors.append(f"{len(dup)} duplicate image_id in predictions: {dup[:10]}")
    if missing:
        errors.append(f"{len(missing)} GT image_id missing from predictions: {missing[:10]}")
    if extra:
        errors.append(f"{len(extra)} predictions have image_id not present in GT: {extra[:10]}")
    if len(pred_ids) != n_expected:
        errors.append(f"prediction count {len(pred_ids)} != expected {n_expected}")
    if errors:
        raise IncompletePredictionsError(
            f"predictions are INCOMPLETE relative to GT -- refusing to score. " + "; ".join(errors)
        )


def score_all(gt_rows: dict[str, GTRow], records: list[dict],
              aliases: dict[str, list[str]] | None = None,
              expected_count: int | None = None) -> dict:
    """Compute the full metric set for one method's prediction records. Records are the
    JSONL dicts written by methods/*.py (MethodResult.to_json_record()).

    Calls assert_complete_predictions() FIRST, unconditionally -- this is the hard guard
    (not just a unit test) that prevents ever silently scoring an incomplete prediction
    set (missing/extra/duplicate image_id relative to GT). Applies to every caller of
    score_all(), not just the CLI below.
    """
    assert_complete_predictions(gt_rows, records, expected_count)

    candidate = _predictions_map(records, "candidate_brands")
    accepted = _predictions_map(records, "accepted_brands")
    released_brands = _predictions_map(records, "released_brands")
    released_products = _predictions_map(records, "released_products")
    rejected = _predictions_map(records, "rejected_brands")
    finals = {d["image_id"]: d.get("final_summary", "") for d in records}

    brand_prf1 = score_entity_prf1(gt_rows, released_brands, "brands", aliases)
    product_prf1 = score_entity_prf1(gt_rows, released_products, "products", aliases)
    ubr = score_ubr(gt_rows, released_brands, aliases)
    cbr = score_cbr(candidate, accepted, gt_rows, aliases)
    rbl = score_rbl(rejected, finals)
    fbrr = score_false_brand_release_rate(gt_rows, released_brands)

    return {
        "brand_prf1": brand_prf1.as_dict(),
        "product_prf1": product_prf1.as_dict(),
        "ubr": ubr,
        "cbr": cbr,
        "rbl": rbl,
        "false_brand_release_rate": fbrr,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=False,
                     help="JSONL of MethodResult.to_json_record() dicts for ONE method")
    ap.add_argument("--gt", type=Path, default=None)
    ap.add_argument("--expected-count", type=int, default=None,
                     help="defaults to the GT row count (582 for the real blind GT); "
                          "an explicit pin catches a wrong/stale --gt file too")
    args = ap.parse_args()
    if not args.predictions:
        print("This is a library module with a CLI stub. Pass --predictions <file.jsonl> "
              "--gt <benchmark_gt_final.csv> once fixed_drafts + method outputs exist. "
              "Not executed in this preparation phase.")
        raise SystemExit(0)
    gt_path = args.gt or (Path(__file__).resolve().parent.parent.parent.parent.parent
                           / "hackathon_benchmark" / "hackathon_benchmark_dataset"
                           / "benchmark_gt_final.csv")
    gt_rows = load_gt(gt_path)
    aliases = load_aliases()
    records = []
    with open(args.predictions, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    try:
        result = score_all(gt_rows, records, aliases, expected_count=args.expected_count)
    except IncompletePredictionsError as exc:
        print(f"REFUSED TO SCORE: {exc}")
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
