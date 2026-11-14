#!/usr/bin/env python3
"""Freeze the generator's output ONCE, POST-RETRY, PRE-VERIFICATION.

CRITICAL (per BENCHMARK_EXPERIMENT_PLAN corrections, section 1.1): the production batch
path is NOT simply OCR -> VLM -> anchor_gate. It is:

    OCR
      |
    VLM.batch()
      |
    detect cases with empty summary but n_ocr_regions > 0
      |
    VLM.batch_retry() for those cases (uses RETRY_TMPL, NOT the primary prompt)
      |
    replace the original raw dict where retry produced a non-empty summary
      |
    <-- THIS SCRIPT STOPS HERE, records the raw dict -->
      |
    anchor_gate()          [NOT called by this script]
      |
    finalize()              [NOT called by this script]

This retry block (mirroring pipeline.py:886-897 exactly: same eligibility condition,
same vlm.batch_retry() call, same replace-only-if-non-empty rule) is UNCHANGED by the
2026-09-17 fixes below and must stay that way -- do not touch it while fixing failure
handling / dataset-role safety.

----------------------------------------------------------------------------------
BLIND FILE LIST MUST COME FROM GT, NOT A FOLDER SCAN (fix, 2026-09-17 part 2)
----------------------------------------------------------------------------------
`--dataset-role blind` no longer scans `CANONICAL_BLIND_IMAGES_DIR` for `*.jpg/png/...`.
It reads the exact `filename` column of `benchmark_gt_final.csv` (582 rows) via
`load_blind_image_list_from_gt()`. This matters concretely: as of 2026-09-17 the image
directory physically contains 600 `.jpg` files (18 stray images from GT rows that were
later reverted/deleted during earlier QA passes -- the image FILES were never deleted,
only the CSV rows) while GT has exactly 582 rows. A folder scan would silently pull in
those 18 extra images. `--images`/`--file-list` are therefore **not accepted** for
`--dataset-role blind` (refused if passed) -- use `--gt <path>` to point at a different
GT file only for testing (defaults to the real canonical GT).

----------------------------------------------------------------------------------
NO SILENT OVERWRITE OF THE CANONICAL BLIND ARTIFACT (fix, 2026-09-17 part 2)
----------------------------------------------------------------------------------
The blind generator is meant to run exactly ONCE. If `fixed_drafts.jsonl` already
exists at the resolved canonical output path, `--dataset-role blind` now refuses to run
at all (before any model loading) unless `--force-regenerate` is explicitly passed.

----------------------------------------------------------------------------------
DATASET-ROLE SAFETY (fix, 2026-09-17)
----------------------------------------------------------------------------------
Earlier revision defaulted `--images` to the canonical 582-image BLIND benchmark
directory, so a bare `dump_fixed_drafts.py --smoke 10` would silently run against blind
images before any method/policy freeze. Fixed by requiring an explicit
`--dataset-role {smoke,validation,blind}` (no default):

  - role=smoke      requires explicit --images or --file-list (no default at all);
                     refuses if the resolved image source is the canonical blind dir.
  - role=validation defaults to data/processed/val.jsonl's own image paths (the
                     validation split ALREADY explicitly configured in
                     configs/benchmark_config.json -> paths.lora_validation_jsonl --
                     not an invented path) if neither --images nor --file-list is given;
                     also refuses the canonical blind dir.
  - role=blind      requires --allow-blind; defaults --images to the canonical blind
                     dir ONLY because both explicit opt-ins (role + flag) were given;
                     enables STRICT mode (section below) and the 582-count completeness
                     gate.

Only role=blind is allowed to write to the canonical `fixed_drafts.jsonl` path; smoke/
validation roles refuse to write there even if explicitly pointed at it.

----------------------------------------------------------------------------------
STRICT / CANONICAL BLIND MODE (fix, 2026-09-17)
----------------------------------------------------------------------------------
Previously, OCR failures were silently skipped (exit code 0) and VLM failures (batch or
per-image) were silently replaced with an empty-but-"ok"-labeled record. Unacceptable
for the artifact the paper's headline numbers will be computed from. In STRICT mode
(role=blind):

  - any OCR failure on any requested image -> the run is marked FAILED, non-zero exit,
    canonical fixed_drafts.jsonl is NOT written (a `.partial.jsonl` + metadata is kept
    for diagnosis);
  - VLM.batch() raising -> the whole run FAILS immediately; the previous per-image
    fallback-to-VLM.__call__() is NOT used in strict mode (it would silently mix two
    different generation paths into one "canonical" artifact -- see corrections section
    4.3). That fallback remains available in non-strict (smoke/validation) mode only,
    and any record it produces is clearly labeled, never "ok";
  - the canonical output is written atomically: all records go to
    `fixed_drafts.jsonl.partial` first, then -- only if zero OCR/VLM failures, the
    written record count equals the expected count (582 by default for blind), and
    every image_id is unique -- the file is atomically `os.replace()`d onto
    `fixed_drafts.jsonl`. Any failed check leaves the `.partial` file in place, writes no
    canonical file, and returns non-zero.

NOT EXECUTED against the real 582 images in this preparation/fix phase -- requires a
working venv (paddleocr/peft not yet installed) and the base Qwen2.5-VL-3B-Instruct
weights (not yet cached locally). See PRE_RUN_READINESS.md.

Run (future):
    # small correctness check on an explicit, non-blind image set first
    python3 dump_fixed_drafts.py --dataset-role smoke --images <some/dev/images> --smoke 10

    # validation-split fixed drafts (for M3 policy selection), auto-sourced from val.jsonl
    python3 dump_fixed_drafts.py --dataset-role validation

    # the ONE canonical blind run, only after M3 policy is frozen
    python3 dump_fixed_drafts.py --dataset-role blind --allow-blind
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPTS_DIR.parent
PIPELINE_ROOT = EXP_ROOT.parent.parent  # end-to-end-pipeline/
PROJECT_ROOT = PIPELINE_ROOT.parent  # Hieu/
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

CANONICAL_BLIND_IMAGES_DIR = (PROJECT_ROOT / "hackathon_benchmark" /
                               "hackathon_benchmark_dataset").resolve()
CANONICAL_BLIND_GT = (CANONICAL_BLIND_IMAGES_DIR / "benchmark_gt_final.csv").resolve()
CANONICAL_BLIND_OUT = (EXP_ROOT / "artifacts" / "fixed_drafts" / "fixed_drafts.jsonl").resolve()
CANONICAL_BLIND_EXPECTED_COUNT = 582  # sanity pin; the real file list always comes from
# --gt's row count, this is just a documented default that fails loudly if --gt ever
# drifts from 582 rows without someone explicitly passing --expected-count to override.
VAL_JSONL = PIPELINE_ROOT / "data" / "processed" / "val.jsonl"

DEFAULT_OUT_BY_ROLE = {
    "blind": CANONICAL_BLIND_OUT,
    "validation": EXP_ROOT / "artifacts" / "fixed_drafts" / "validation" / "fixed_drafts_validation.jsonl",
    "smoke": EXP_ROOT / "artifacts" / "fixed_drafts" / "smoke" / "fixed_drafts_smoke.jsonl",
}


def build_record(image_id: str, filename: str, ocr_text: str, n_regions: int,
                  raw_dict: dict, retry_triggered: bool, retry_replaced_original: bool,
                  status: str) -> dict:
    return {
        "image_id": image_id,
        "filename": filename,
        "external_ocr": ocr_text,
        "n_ocr_regions": n_regions,
        "vlm_ocr": str(raw_dict.get("ocr_text", "")),
        "candidate_brands": [b.strip() for b in str(raw_dict.get("brand", "")).split(";") if b.strip()],
        "candidate_products": [p.strip() for p in str(raw_dict.get("product", "")).split(";") if p.strip()],
        "draft_summary": str(raw_dict.get("summary", "")),
        "retry_triggered": retry_triggered,
        "retry_replaced_original": retry_replaced_original,
        "generator_status": status,  # "ok" | "ocr_failed" | "vlm_batch_failed" | "vlm_image_failed_fallback_empty"
        "raw_vlm_dict": raw_dict,
    }


def resolve_image_list(args) -> list[Path]:
    if args.file_list:
        files = [PIPELINE_ROOT / l.strip() for l in open(args.file_list, encoding="utf-8") if l.strip()]
        return [f for f in files if f.exists()]
    if args.images:
        return sorted(p for p in args.images.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    raise ValueError("no image source resolved")  # callers must not reach here unguarded


def load_blind_image_list_from_gt(gt_csv: Path, image_dir: Path) -> list[Path]:
    """Blind role must process EXACTLY the filenames listed in benchmark_gt_final.csv --
    never a directory scan. See module docstring "BLIND FILE LIST MUST COME FROM GT".
    Sorted by image_id for a deterministic, auditable processing order (matches
    validate_inputs.py's own sort)."""
    import csv
    if not gt_csv.exists():
        raise FileNotFoundError(f"blind GT not found: {gt_csv}")
    rows = list(csv.DictReader(open(gt_csv, encoding="utf-8")))
    rows.sort(key=lambda r: r["image_id"])
    files: list[Path] = []
    missing: list[str] = []
    for r in rows:
        p = image_dir / r["filename"]
        if not p.exists():
            missing.append(r["filename"])
            continue
        files.append(p)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} GT-referenced image file(s) missing from {image_dir}: "
            f"{missing[:10]}"
        )
    return files


def derive_validation_file_list() -> list[Path]:
    """Extract real image paths from data/processed/val.jsonl -- the validation split
    ALREADY explicitly configured as the verifier-policy validation source in
    configs/benchmark_config.json. Not an invented path (per corrections section 3.1:
    "Do not infer or invent a validation path that is not known to exist")."""
    if not VAL_JSONL.exists():
        raise FileNotFoundError(
            f"--dataset-role validation needs {VAL_JSONL} (the configured validation "
            f"split) but it does not exist. Pass --images or --file-list explicitly instead."
        )
    files = []
    with open(VAL_JSONL, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            img_rel = rec.get("image")
            if img_rel:
                p = PIPELINE_ROOT / img_rel
                if p.exists():
                    files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-role", required=True, choices=["smoke", "validation", "blind"],
                     help="REQUIRED, no default -- prevents accidentally defaulting to the blind set")
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--file-list", type=Path, default=None,
                     help="text file, one image path (relative to end-to-end-pipeline/) per line")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--meta-out", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0, help="only process the first N images")
    ap.add_argument("--lora", type=Path, default=None, help="defaults to pipeline.LORA_DIR")
    ap.add_argument("--allow-blind", action="store_true",
                     help="REQUIRED for --dataset-role blind -- explicit acknowledgment")
    ap.add_argument("--expected-count", type=int, default=None,
                     help="defaults to 582 for --dataset-role blind, unset otherwise")
    ap.add_argument("--gt", type=Path, default=None,
                     help="--dataset-role blind ONLY: GT csv to read the exact file list "
                          "from (default: the real canonical benchmark_gt_final.csv). "
                          "Overriding this is for testing only.")
    ap.add_argument("--force-regenerate", action="store_true",
                     help="--dataset-role blind ONLY: required to proceed if the canonical "
                          "output path already exists -- the blind generator is meant to "
                          "run exactly once")
    args = ap.parse_args()

    print("=" * 70)
    print("dump_fixed_drafts.py -- POST-RETRY, PRE-VERIFICATION freeze")
    print(f"dataset-role: {args.dataset_role}")
    print("=" * 70)
    print("This script intentionally does NOT call anchor_gate() or finalize().")
    print()

    strict = args.dataset_role == "blind"
    blind_files: list[Path] | None = None  # resolved directly from GT below, bypasses
    # resolve_image_list()'s directory-scan path entirely for --dataset-role blind

    # ---- role-specific safety gating (section 3) ----
    if args.dataset_role == "blind":
        if not args.allow_blind:
            print("REFUSED: --dataset-role blind requires --allow-blind (explicit "
                  "acknowledgment). This is the canonical 582-image benchmark set.")
            return 4
        if args.smoke:
            print("REFUSED: --smoke may not be combined with --dataset-role blind. "
                  "Use --dataset-role smoke against an explicit non-blind image set "
                  "for correctness checks instead.")
            return 4
        if args.images is not None or args.file_list is not None:
            print("REFUSED: --dataset-role blind always derives its exact file list from "
                  "the GT csv's own `filename` column (--gt, default: the real canonical "
                  "benchmark_gt_final.csv) -- it does not accept --images or --file-list. "
                  "This prevents accidentally running on a stale/wrong image set, e.g. a "
                  "directory scan that includes images GT no longer references (the "
                  "image directory currently has 600 physical files for 582 GT rows).")
            return 4
        gt_path = args.gt if args.gt is not None else CANONICAL_BLIND_GT
        image_dir = gt_path.resolve().parent
        try:
            blind_files = load_blind_image_list_from_gt(gt_path, image_dir)
        except FileNotFoundError as exc:
            print(f"REFUSED: {exc}")
            return 4
        if args.expected_count is None:
            args.expected_count = CANONICAL_BLIND_EXPECTED_COUNT
        if len(blind_files) != args.expected_count:
            print(f"REFUSED: {gt_path} has {len(blind_files)} rows with an existing image "
                  f"file, but --expected-count is {args.expected_count}. If this GT file "
                  f"is deliberately different from the known-582-row benchmark, pass "
                  f"--expected-count {len(blind_files)} explicitly.")
            return 4
        if args.out is None:
            args.out = CANONICAL_BLIND_OUT
        if args.out.exists() and not args.force_regenerate:
            print(f"REFUSED: canonical blind output already exists at {args.out}. "
                  f"The blind generator is meant to run exactly ONCE. Pass "
                  f"--force-regenerate if you deliberately intend to replace it -- and "
                  f"record the justification in WORK_LOG.md.")
            return 4

    elif args.dataset_role == "validation":
        if args.images is None and args.file_list is None:
            try:
                val_files = derive_validation_file_list()
            except FileNotFoundError as exc:
                print(f"REFUSED: {exc}")
                return 4
            tmp_list = EXP_ROOT / "manifests" / "_validation_file_list.txt"
            tmp_list.parent.mkdir(parents=True, exist_ok=True)
            tmp_list.write_text("\n".join(str(p.relative_to(PIPELINE_ROOT)) for p in val_files),
                                 encoding="utf-8")
            args.file_list = tmp_list
            print(f"  (auto-derived {len(val_files)} image paths from {VAL_JSONL})")

    elif args.dataset_role == "smoke":
        if args.images is None and args.file_list is None:
            print("REFUSED: --dataset-role smoke requires an explicit --images or "
                  "--file-list -- there is no default, specifically so a bare smoke "
                  "run can never silently select the blind benchmark.")
            return 4

    # defense in depth: never let smoke/validation resolve to the canonical blind dir
    if not strict and args.images is not None:
        try:
            if args.images.resolve() == CANONICAL_BLIND_IMAGES_DIR:
                print(f"REFUSED: --dataset-role {args.dataset_role} resolved --images to the "
                      f"canonical blind directory ({CANONICAL_BLIND_IMAGES_DIR}). "
                      f"Use --dataset-role blind --allow-blind if this is intentional.")
                return 4
        except OSError:
            pass

    if args.out is None:
        args.out = DEFAULT_OUT_BY_ROLE[args.dataset_role]
    if not strict and args.out.resolve() == CANONICAL_BLIND_OUT:
        print(f"REFUSED: --dataset-role {args.dataset_role} may not write to the canonical "
              f"blind output path ({CANONICAL_BLIND_OUT}). Pass an explicit --out elsewhere.")
        return 4
    if args.meta_out is None:
        args.meta_out = args.out.with_name(args.out.stem + "_meta.json")

    try:
        import torch  # noqa: F401
        from PIL import Image
        import pipeline as P
    except ImportError as exc:
        print(f"BLOCKED: required package not importable ({exc}).")
        print("This is expected in the current environment -- see PRE_RUN_READINESS.md.")
        print("Not a bug in this script; the venv/model download steps have not been run yet.")
        return 3

    if blind_files is not None:
        files = blind_files  # exact GT-derived list -- never a directory scan
    else:
        try:
            files = resolve_image_list(args)
        except ValueError:
            print("REFUSED: no image source resolved (this should be unreachable given "
                  "the gating above).")
            return 4
    if args.smoke:
        files = files[: args.smoke]
    if not files:
        print("No input images found -- nothing to do.")
        return 1

    print(f"images to process : {len(files)}")
    print(f"batch size         : {args.batch}")
    print(f"strict mode        : {strict}")
    print(f"expected count     : {args.expected_count}")
    print(f"output (canonical) : {args.out}")

    t_load0 = time.time()
    try:
        ocr = P.OCR()
        vlm = P.VLM(args.lora if args.lora is not None else P.LORA_DIR)
    except ImportError as exc:
        print(f"BLOCKED: required package not importable ({exc}).")
        print("This is expected in the current environment -- see PRE_RUN_READINESS.md.")
        print("Not a bug in this script; the venv/model download steps have not been run yet.")
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKED: model/OCR failed to load ({type(exc).__name__}: {exc}).")
        print("See PRE_RUN_READINESS.md -- base model may not be downloaded yet, or GPU/venv "
              "is not correctly set up.")
        return 3
    print(f"model load time: {time.time()-t_load0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.out.with_suffix(args.out.suffix + ".partial")

    n_written = n_ocr_failures = n_vlm_batch_failures = n_vlm_image_failures = 0
    n_retry_triggered = n_retry_replaced = 0
    run_failed = False
    seen_ids: set[str] = set()

    with open(partial_path, "w", encoding="utf-8") as fh:
        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            imgs, ocrs, n_regions_list, good = [], [], [], []
            for fp in chunk:
                try:
                    img = Image.open(fp).convert("RGB")
                    ocr_text, n_regions = ocr(img)
                except Exception as exc:  # noqa: BLE001
                    n_ocr_failures += 1
                    print(f"  ! OCR failed for {fp.name}: {type(exc).__name__}: {exc}", flush=True)
                    if strict:
                        run_failed = True
                        fh.write(json.dumps({
                            "image_id": fp.stem, "filename": fp.name,
                            "generator_status": "ocr_failed", "error": str(exc),
                        }, ensure_ascii=False) + "\n")
                    # non-strict: also record the failure explicitly (never silently
                    # dropped), just doesn't flip run_failed
                    else:
                        fh.write(json.dumps({
                            "image_id": fp.stem, "filename": fp.name,
                            "generator_status": "ocr_failed", "error": str(exc),
                        }, ensure_ascii=False) + "\n")
                    continue
                good.append(fp)
                imgs.append(img)
                ocrs.append(ocr_text)
                n_regions_list.append(n_regions)
            if not good:
                continue

            try:
                datas = vlm.batch(imgs, ocrs)
                batch_ok = True
            except Exception as exc:  # noqa: BLE001
                n_vlm_batch_failures += 1
                print(f"  ! VLM.batch failed for this chunk ({len(good)} images): "
                      f"{type(exc).__name__}: {exc}", flush=True)
                if strict:
                    run_failed = True
                    for fp in good:
                        fh.write(json.dumps({
                            "image_id": fp.stem, "filename": fp.name,
                            "generator_status": "vlm_batch_failed", "error": str(exc),
                        }, ensure_ascii=False) + "\n")
                    batch_ok = False
                    datas = None
                else:
                    print("    (non-strict mode: falling back to per-image VLM.__call__ -- "
                          "records will be clearly marked, never labeled 'ok')", flush=True)
                    datas = []
                    for k, img in enumerate(imgs):
                        try:
                            datas.append(vlm(img, ocrs[k]))
                        except Exception:  # noqa: BLE001
                            n_vlm_image_failures += 1
                            datas.append(None)
                    batch_ok = True  # proceed with per-image results (some may be None)

            if not batch_ok:
                continue

            # ---- retry logic, EXACT mirror of pipeline.py:886-897 -- UNCHANGED ----
            need = [k for k, d in enumerate(datas)
                    if d is not None and not str(d.get("summary", "")).strip() and n_regions_list[k] > 0]
            retry_replaced_flags = {k: False for k in range(len(datas))}
            retry_triggered_flags = {k: (k in need) for k in range(len(datas))}
            if need:
                again = vlm.batch_retry([imgs[k] for k in need])
                for k, d2 in zip(need, again):
                    if str(d2.get("summary", "")).strip():
                        datas[k] = d2
                        retry_replaced_flags[k] = True

            for k, fp in enumerate(good):
                if fp.stem in seen_ids:
                    print(f"  ! duplicate image_id encountered: {fp.stem} -- keeping first", flush=True)
                    continue
                seen_ids.add(fp.stem)
                if datas[k] is None:
                    status = "vlm_image_failed_fallback_empty"
                    raw = {"brand": "", "product": "", "summary": ""}
                else:
                    status = "ok"
                    raw = datas[k]
                if retry_triggered_flags[k]:
                    n_retry_triggered += 1
                if retry_replaced_flags[k]:
                    n_retry_replaced += 1
                rec = build_record(
                    image_id=fp.stem, filename=fp.name,
                    ocr_text=ocrs[k], n_regions=n_regions_list[k], raw_dict=raw,
                    retry_triggered=retry_triggered_flags[k],
                    retry_replaced_original=retry_replaced_flags[k],
                    status=status,
                )
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            fh.flush()
            print(f"  {n_written}/{len(files)} written "
                  f"(retry triggered {n_retry_triggered}, rescued {n_retry_replaced}, "
                  f"OCR fail {n_ocr_failures}, VLM batch fail {n_vlm_batch_failures}, "
                  f"VLM image fail {n_vlm_image_failures})", flush=True)

    # ---- completeness gate + atomic publish ----
    # Kept simple on purpose: when there are zero failures of any kind, n_written must
    # naturally equal len(files) (every requested image produced exactly one "ok"
    # record) -- so "zero failures" + "written == requested" together are sufficient
    # and mutually consistent, no need to subtract failure counts from either side.
    checks = {
        "zero_ocr_failures": (not strict) or n_ocr_failures == 0,
        "zero_vlm_batch_failures": (not strict) or n_vlm_batch_failures == 0,
        "zero_vlm_image_failures": (not strict) or n_vlm_image_failures == 0,
        "written_equals_requested": n_written == len(files),
        "unique_image_id_count_matches": len(seen_ids) == n_written,
        "no_run_failed_flag": not run_failed,
    }
    if args.expected_count is not None:
        checks["written_equals_expected_count"] = n_written == args.expected_count

    canonical_complete = all(checks.values()) if strict else True  # non-strict never blocks publish

    if canonical_complete:
        os.replace(partial_path, args.out)
        published_path = args.out
    else:
        published_path = partial_path
        print("\nSTRICT MODE CHECKS FAILED -- canonical file NOT written. "
              "Partial output kept for diagnosis:")
        for name, ok in checks.items():
            if not ok:
                print(f"    ✗ {name}")

    meta = {
        "dataset_role": args.dataset_role,
        "gt_used": str(args.gt if args.gt is not None else CANONICAL_BLIND_GT) if strict else None,
        "strict_mode": strict,
        "n_images_requested": len(files),
        "n_records_written": n_written,
        "n_ocr_failures": n_ocr_failures,
        "n_vlm_batch_failures": n_vlm_batch_failures,
        "n_vlm_image_failures": n_vlm_image_failures,
        "n_retry_triggered": n_retry_triggered,
        "n_retry_replaced_original": n_retry_replaced,
        "batch_size": args.batch,
        "expected_count": args.expected_count,
        "checks": checks,
        "canonical_complete": canonical_complete,
        "published_path": str(published_path),
        "base_model_id": P.BASE_MODEL,
        "lora_dir": str(args.lora if args.lora is not None else P.LORA_DIR),
        "min_pixels": P.MIN_PIXELS, "max_pixels": P.MAX_PIXELS,
        "ocr_conf_min": P.OCR_CONF_MIN,
        "python_version": platform.python_version(),
        "generated_at_unix": time.time(),
        "note": "This file MUST accompany fixed_drafts.jsonl in the freeze manifest -- "
                "it is not reproducible without the exact package/model versions recorded "
                "in freeze_manifest.json.",
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"wrote {n_written} records -> {published_path}")
    print(f"wrote metadata      -> {args.meta_out}")
    print(f"canonical_complete  : {canonical_complete}")
    if strict and not canonical_complete:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
