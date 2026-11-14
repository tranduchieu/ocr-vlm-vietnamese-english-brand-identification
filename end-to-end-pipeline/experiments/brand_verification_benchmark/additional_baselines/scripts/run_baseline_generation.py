#!/usr/bin/env python3
"""Generation driver for B0 / B1 / L0 (Final Simple Baseline Benchmark).

Reuses the EXACT production VLM class (src/pipeline.py) unmodified -- same
prompt (SYSTEM/USER_TMPL/RETRY_TMPL), same decoding config (do_sample=False,
max_new_tokens=200), same min/max pixels, same retry logic (VLM.batch_retry()
for empty-summary-but-detected-text cases). The ONLY per-system differences are
(a) whether lora is passed to VLM(), and (b) what ocr_text is passed per image --
both config, not new code.

Image list and external OCR are read from the IMMUTABLE Batch C canonical
artifacts/fixed_drafts/fixed_drafts.jsonl (582 rows) -- never a folder scan,
never re-running OCR. Same 582 image IDs as Batch C, by construction.

Writes atomically: .partial -> verify (582/582, unique, no missing/duplicate) ->
os.replace(). Never silently skips a failed image.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = EXP_ROOT.parent.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))
import pipeline as P  # noqa: E402
from PIL import Image  # noqa: E402

FREEZE = json.loads((EXP_ROOT / "additional_baselines" / "configs" / "baseline_freeze.json").read_text(encoding="utf-8"))
CANONICAL_DRAFTS = EXP_ROOT / "artifacts" / "fixed_drafts" / "fixed_drafts.jsonl"
IMAGES_DIR = PROJECT_ROOT / "hackathon_benchmark" / "hackathon_benchmark_dataset"
OUT_DIR = EXP_ROOT / "additional_baselines" / "artifacts" / "fixed_drafts"

SYSTEMS = FREEZE["systems"]


def load_canonical_rows(drafts_path: Path | None = None) -> list[dict]:
    path = drafts_path or CANONICAL_DRAFTS
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(rows) == 582, f"expected 582 canonical rows, got {len(rows)}"
    return rows


def as_text(value) -> str:
    """The base (non-fine-tuned) Qwen model does not always follow the string-only
    schema the production LoRA model reliably does -- observed producing a JSON list
    for 'product'/'brand' where a string was expected (still syntactically valid
    JSON, just a different type). Coerce defensively rather than crash; a list of
    strings is joined with '; ' (the same multi-entity delimiter convention used
    throughout the project), any other non-string is stringified. This is a parsing
    robustness fix, not a prompt/result tuning change -- no scoring logic is touched
    and the raw model output is preserved unmodified in raw_vlm_dict for audit."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value) if value is not None else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["B0", "B1", "L0"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=None, help="only process first N images")
    ap.add_argument(
        "--canonical-drafts",
        type=Path,
        default=None,
        help="override Batch-C fixed_drafts.jsonl (image list + external OCR for B1)",
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="root directory for draft filename paths (default: hackathon benchmark images)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="override additional_baselines/artifacts/fixed_drafts output root",
    )
    args = ap.parse_args()

    canonical_drafts = args.canonical_drafts or CANONICAL_DRAFTS
    images_dir = args.images_dir or IMAGES_DIR
    out_root = args.out_dir or OUT_DIR

    cfg = SYSTEMS[args.system]
    lora_dir = Path(PIPELINE_ROOT / cfg["lora_dir"]) if cfg["lora_dir"] else None

    # HARD SAFETY ASSERTION per section 4 / 13
    if args.system in ("B0", "B1"):
        assert lora_dir is None, f"{args.system} must NOT load LoRA, got lora_dir={lora_dir}"
    else:
        assert lora_dir is not None and lora_dir.exists(), f"L0 must load LoRA from {lora_dir}"

    rows = load_canonical_rows(canonical_drafts)
    if args.smoke:
        rows = rows[: args.smoke]
    total_target = len(rows)  # fixed target count (582, or --smoke N) -- independent
    # of how many rows remain to be processed after a resume filters out already-done
    # ones below; all completeness checks/prints must compare against THIS, not the
    # shrinking `rows` list.

    print(f"system              : {args.system} ({cfg['description']})")
    print(f"lora_dir requested  : {lora_dir}")
    print(f"ocr_mode            : {cfg['ocr_mode']}")
    print(f"images to process   : {len(rows)}")
    print(f"batch size          : {args.batch}")

    t0 = time.time()
    vlm = P.VLM(lora_dir)
    adapter_loaded = lora_dir is not None and lora_dir.exists()
    print(f"model load time: {time.time()-t0:.1f}s")
    print(f"ADAPTER_LOADED      : {adapter_loaded}")
    print(f"MODEL_CLASS         : {type(vlm.model).__name__}")
    print(f"DTYPE               : bfloat16")
    print(f"BASE_MODEL          : {P.BASE_MODEL}")

    if args.system in ("B0", "B1"):
        assert not adapter_loaded, "SAFETY VIOLATION: B0/B1 accidentally loaded LoRA"
    else:
        assert adapter_loaded, "SAFETY VIOLATION: L0 failed to load LoRA"

    out_name = f"fixed_drafts_{args.system.lower()}" + (f"_smoke{args.smoke}" if args.smoke else "")
    out_path = out_root / args.system.lower() / f"{out_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")

    n_written = n_retry_triggered = n_retry_replaced = 0
    seen_ids = set()

    # RESUME: if a .partial file already exists from a previous OOM'd attempt, keep
    # its already-written records (each was a genuine, complete, successful generation
    # -- not a partial/truncated record) and only regenerate the remaining images. This
    # is purely an efficiency optimization on retry (avoids redoing already-successful
    # work after a resource-contention OOM) -- every image still gets generated exactly
    # once for real, none silently skipped; final completeness is still verified below.
    already_done = []
    if partial_path.exists():
        for line in open(partial_path, encoding="utf-8"):
            if line.strip():
                already_done.append(json.loads(line))
        seen_ids = {r["image_id"] for r in already_done}
        n_written = len(already_done)
        print(f"RESUMING from existing partial: {n_written} already-written records kept")
        rows = [r for r in rows if r["image_id"] not in seen_ids]

    file_mode = "a" if already_done else "w"
    with open(partial_path, file_mode, encoding="utf-8") as fh:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            imgs, ocr_texts, n_regions_list = [], [], []
            for row in chunk:
                img_path = images_dir / row["filename"]
                img = Image.open(img_path).convert("RGB")
                imgs.append(img)
                if cfg["ocr_mode"].startswith("canonical"):
                    ocr_texts.append(row["external_ocr"])
                    n_regions_list.append(row["n_ocr_regions"])
                else:
                    ocr_texts.append("")
                    n_regions_list.append(0)  # no external OCR evidence for this system

            datas = vlm.batch(imgs, ocr_texts)

            # retry logic -- EXACT mirror of pipeline.py:886-897 / dump_fixed_drafts.py
            need = [k for k, d in enumerate(datas)
                    if d is not None and not str(d.get("summary", "")).strip() and n_regions_list[k] > 0]
            retry_triggered = {k: (k in need) for k in range(len(datas))}
            retry_replaced = {k: False for k in range(len(datas))}
            if need:
                again = vlm.batch_retry([imgs[k] for k in need])
                for k, d2 in zip(need, again):
                    if str(d2.get("summary", "")).strip():
                        datas[k] = d2
                        retry_replaced[k] = True

            for k, row in enumerate(chunk):
                iid = row["image_id"]
                if iid in seen_ids:
                    print(f"  ! duplicate image_id {iid} -- keeping first", flush=True)
                    continue
                seen_ids.add(iid)
                raw = datas[k] or {"brand": "", "product": "", "summary": ""}
                if retry_triggered[k]:
                    n_retry_triggered += 1
                if retry_replaced[k]:
                    n_retry_replaced += 1
                rec = {
                    "image_id": iid, "filename": row["filename"],
                    "external_ocr": ocr_texts[k], "n_ocr_regions": n_regions_list[k],
                    "vlm_ocr": as_text(raw.get("ocr_text", "")),
                    "candidate_brands": P.split_multi(as_text(raw.get("brand", ""))),
                    "candidate_products": P.split_multi(as_text(raw.get("product", ""))),
                    "draft_summary": raw.get("summary", ""),
                    "retry_triggered": retry_triggered[k], "retry_replaced_original": retry_replaced[k],
                    "generator_status": "ok" if datas[k] is not None else "vlm_image_failed_fallback_empty",
                    "raw_vlm_dict": raw,
                    "system": args.system, "model_id": P.BASE_MODEL, "adapter_loaded": adapter_loaded,
                    "prompt_hash": FREEZE["shared"]["prompt_hash"],
                    "generation_config_hash": FREEZE["shared"]["generation_config_hash"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            print(f"  {n_written}/{total_target} written (retry triggered {n_retry_triggered}, "
                  f"replaced {n_retry_replaced})", flush=True)

    complete = (n_written == total_target) and (len(seen_ids) == total_target)
    if complete:
        os.replace(partial_path, out_path)
        print(f"\nwrote {n_written} records -> {out_path}")
    else:
        print(f"\nINCOMPLETE: {n_written}/{total_target} written, {len(seen_ids)} unique -- "
              f"partial file kept at {partial_path}, NOT published.")
        return 2

    meta = {
        "system": args.system, "n_requested": total_target, "n_written": n_written,
        "n_unique": len(seen_ids), "n_retry_triggered": n_retry_triggered,
        "n_retry_replaced": n_retry_replaced, "batch_size": args.batch,
        "adapter_loaded": adapter_loaded, "model_id": P.BASE_MODEL,
        "lora_dir": str(lora_dir) if lora_dir else None,
        "prompt_hash": FREEZE["shared"]["prompt_hash"],
        "generation_config_hash": FREEZE["shared"]["generation_config_hash"],
        "generated_at_unix": time.time(), "complete": complete,
    }
    (out_path.parent / f"{out_name}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
