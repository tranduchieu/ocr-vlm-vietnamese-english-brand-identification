#!/usr/bin/env bash
# Full paper Table V rerun (B0–M3) on reviewed benchmark GT + public images.
set -euo pipefail

ROOT="/home/tinnl/Giaitri/AI/Projects/Hieu"
E2E="$ROOT/end-to-end-pipeline"
VENV="$E2E/venv/bin/python3"
GT="$ROOT/paper_dataset_release/brand_dataset/benchmark/benchmark_gt_scoring.csv"
BENCH_DIR="$ROOT/paper_dataset_release/brand_dataset/benchmark"
OUT="$E2E/outputs/paper_rerun_20261113"
LOG="$OUT/run.log"
DUMP="$OUT/fixed_drafts_m0.jsonl"
DRAFTS_B="$OUT/fixed_drafts_baselines"
PRED="$OUT/predictions"

mkdir -p "$OUT" "$DRAFTS_B" "$PRED"
exec > >(tee -a "$LOG") 2>&1

echo "=== paper benchmark full $(date -Iseconds) ==="
cd "$E2E"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

score_baselines() {
  local sys="$1"
  local draft="$DRAFTS_B/${sys,,}/fixed_drafts_${sys,,}.jsonl"
  if [[ ! -f "$draft" ]]; then
    echo "missing draft $draft"
    return 1
  fi
  "$VENV" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("$E2E/experiments/brand_verification_benchmark")))
from methods import m0_no_verification as m0
drafts = [json.loads(l) for l in open("$draft", encoding="utf-8") if l.strip()]
out = Path("$PRED") / "${sys,,}.jsonl"
with out.open("w", encoding="utf-8") as fh:
    for d in drafts:
        r = m0.run(
            image_id=d["image_id"],
            candidate_brands=d.get("candidate_brands", []),
            candidate_products=d.get("candidate_products", []),
            draft_summary=d.get("draft_summary", ""),
            external_ocr=d.get("external_ocr", ""),
            vlm_ocr=d.get("vlm_ocr", ""),
            n_ocr_lines=d.get("n_ocr_regions", 0),
        )
        rec = r.to_json_record()
        rec["system"] = "$sys"
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", len(drafts), "->", out)
PY
}

run_step() {
  local name="$1"
  shift
  echo "--- $name ---"
  if "$@"; then
    echo "OK $name"
  else
    echo "FAIL $name (exit $?)"
    exit 1
  fi
}

if [[ ! -f "$DUMP" ]]; then
  run_step "dump M0 drafts (LoRA + OCR)" \
    "$VENV" experiments/brand_verification_benchmark/scripts/dump_fixed_drafts.py \
      --dataset-role blind --allow-blind --force-regenerate \
      --gt "$GT" \
      --out "$DUMP" \
      --meta-out "$OUT/fixed_drafts_m0_meta.json" \
      --batch 8
else
  echo "skip dump (exists $DUMP)"
fi

run_step "verifiers M0–M3" \
  "$VENV" experiments/brand_verification_benchmark/scripts/run_verifiers.py \
    --drafts "$DUMP" \
    --out-dir "$PRED" \
    --methods m0 m1 m2 m3

BASE_GEN="$E2E/experiments/brand_verification_benchmark/additional_baselines/scripts/run_baseline_generation.py"

for SYS in B0 B1 L0; do
  draft="$DRAFTS_B/${SYS,,}/fixed_drafts_${SYS,,}.jsonl"
  if [[ -f "$draft" ]]; then
    echo "skip $SYS generation (exists)"
  else
    run_step "generate $SYS" \
      "$VENV" "$BASE_GEN" \
        --system "$SYS" \
        --batch 8 \
        --canonical-drafts "$DUMP" \
        --images-dir "$BENCH_DIR/images" \
        --out-dir "$DRAFTS_B"
  fi
  run_step "score $SYS via m0 finalize" score_baselines "$SYS"
done

run_step "assemble Table V" \
  "$VENV" scripts/assemble_table_v.py \
    --gt "$GT" \
    --pred-dir "$PRED" \
    --out "$OUT/table_v.json"

"$VENV" scripts/compute_paper_dataset_stats.py
touch "$OUT/COMPLETE"
echo "=== DONE $(date -Iseconds) ==="
