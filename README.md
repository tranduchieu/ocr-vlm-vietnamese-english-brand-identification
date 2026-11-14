# OCR-assisted brand identification (reproduction package)

**Not included (Zenodo):** images/labels (`brand_dataset/`) and LoRA weights.  
**Dataset DOI:** https://doi.org/10.5281/zenodo.24212709  
Optional LoRA archive: `lora_qwen25vl3b_brand.zip` on the same record (if uploaded).

## Layout after downloads

```
tanahi_ocr_vlm/
  dataset/brand_dataset/          # dev/ + benchmark/ from Zenodo
  end-to-end-pipeline/
    models/qwen25vl3b-lora/       # adapter from LoRA zip
    requirements.txt
    src/pipeline.py
  benchmark/gt_benchmark_582.csv  # scoring GT (bundled)
  benchmark/table_v_paper.json    # paper Table V reference
```

## Environment

- Python 3.10+, NVIDIA GPU (~16 GB VRAM for batch 8 on 3B VLM)
- Base model `Qwen/Qwen2.5-VL-3B-Instruct` (Hugging Face, downloaded on first run)

```bash
cd end-to-end-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
```

## Reproduce Table V (B0–M3, 582 images)

```bash
cd end-to-end-pipeline
bash scripts/run_paper_benchmark_full.sh
diff -q outputs/paper_benchmark/table_v.json ../benchmark/table_v_paper.json || \
  python3 -c "import json; a=json.load(open('outputs/paper_benchmark/table_v.json')); b=json.load(open('../benchmark/table_v_paper.json')); print('ok' if a==b else a,b)"
```

## Run on one image

```bash
python src/pipeline.py --image /path/to/image.jpg
```

## Retrain LoRA (optional)

Uses `data/processed/train.jsonl` and `val.jsonl` (paths point to `../../dataset/brand_dataset/`).

```bash
python src/finetune_student.py --smoke   # pipeline check
python src/finetune_student.py            # full train
```

**Citation:** Tran et al., FJCAI 2027 (brand identification benchmark). Contact: qttho@hcmut.edu.vn
