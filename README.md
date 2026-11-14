# OCR-Assisted Vision-Language Models for Vietnamese–English Brand Identification

This repository contains the source code for the paper *"OCR-Assisted Vision-Language Models for Vietnamese–English Brand Identification: A Benchmark and Empirical Study"* (submitted to FJCAI 2027). It provides the end-to-end inference pipeline, the evaluation scripts used to reproduce the reported results, and the training script for task-specific adaptation.

## Overview

The pipeline identifies brand and product entities in commercial images by combining:

- **External OCR:** PaddleOCR for text detection and VietOCR for text recognition
- **A task-adapted vision-language model:** Qwen2.5-VL-3B-Instruct with a LoRA adapter
- **Lightweight evidence processing:** verification of predicted brands against recognized text

## Data and Model Weights

The dataset is not included in this repository. It is available on Zenodo:

- **Dataset:** [https://doi.org/10.5281/zenodo.24212709](https://doi.org/10.5281/zenodo.24212709)

- **LoRA adapter:** `lora_qwen25vl3b_brand.zip`, available on the same Zenodo record. Extract it to `end-to-end-pipeline/models/qwen25vl3b-lora/`.

The base model, `Qwen/Qwen2.5-VL-3B-Instruct`, is downloaded automatically from Hugging Face on first use. The adapter can also be reproduced with the training script described below.

## Repository Structure

After downloading the dataset and the adapter, the project is organized as follows:

```
tanahi_ocr_vlm/
├── dataset/brand_dataset/          # Dataset from Zenodo
├── end-to-end-pipeline/
│   ├── models/qwen25vl3b-lora/     # LoRA adapter
│   ├── scripts/                    # Benchmark scripts
│   ├── src/                        # Pipeline and training code
│   └── requirements.txt
└── benchmark/                      # Evaluation ground truth and reference results
```

## Installation

**Requirements:** Python 3.10 or later and an NVIDIA GPU with about 16 GB of memory.

```bash
cd end-to-end-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Inference on a single image

```bash
python src/pipeline.py --image path/to/image.jpg
```

The pipeline outputs the recognized text, identified brands and products, and a short image summary in structured form.

### Reproducing the benchmark results

```bash
bash scripts/run_paper_benchmark_full.sh
```

This script runs all experimental configurations reported in the paper on the 582-image benchmark and writes the results to `outputs/paper_benchmark/`. Reference results from the paper are provided in `benchmark/table_v_paper.json` for comparison.

### Training the LoRA adapter

```bash
python src/finetune_student.py
```

The training script uses the training and validation splits of the development collection.

## Citation

If you use this code or the dataset, please cite:

```bibtex
@inproceedings{tran2027brandidentification,
  title     = {OCR-Assisted Vision-Language Models for Vietnamese--English Brand Identification: A Benchmark and Empirical Study},
  author    = {Tran, Hieu D. and Phan, Tai T. and Nguyen, Ngan H. and Le, Nhi N. T. and Duong, Nam H. and Le, Thi-Hong-Cuc and Vo, Khuong D. and Quan, Tho T.},
  booktitle = {Proceedings of the FISU Joint Conference on Artificial Intelligence (FJCAI)},
  year      = {2027}
}
```

## License

- **Source code:** [MIT License](LICENSE)
- **Dataset:** Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
- **Base model and LoRA adapter:** The base model, Qwen2.5-VL-3B-Instruct, is distributed under the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE). The LoRA adapter is derived from this model and is subject to the same terms, which restrict use to non-commercial research. Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.

## Contact

For questions about the code, please contact Hieu D. Tran (hieu.tranduc1031@hcmut.edu.vn).
