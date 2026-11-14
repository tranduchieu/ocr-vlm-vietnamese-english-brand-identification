#!/usr/bin/env python3
"""Giai đoạn 3 — huấn luyện student Qwen2.5-VL-3B bằng LoRA trên nhãn teacher.

VÌ SAO VIẾT LẠI THAY VÌ DÙNG src/finetune.py CŨ:
File cũ dùng Unsloth + `Qwen3-VL-8B-Instruct`. Hai vấn đề chí mạng:
  · 8B PHẠM QUY — đề giới hạn mỗi model đơn lẻ ≤4B (mục 4). Kể cả Qwen3-VL-4B cũng
    phạm quy vì đếm cả vision encoder là 4,438B (đo bằng safetensors, xem WORK_LOG).
  · Unsloth không có trong venv nào của dự án, và nó ghim phiên bản transformers riêng.
Ở đây dùng thẳng transformers + peft — vòng lặp tự viết, không phụ thuộc API của TRL vốn
đổi liên tục giữa các bản. Chạy không người trông thì tính dự đoán được quan trọng hơn
sự tiện tay.

THIẾT KẾ:
  · Student sinh JSON gọn {"brand","product","summary"} chứ không chỉ summary.
    Cần brand/product tách bạch thì CỔNG NEO ở khâu suy luận mới chặn được bịa (Đ1).
  · ĐÓNG BĂNG vision encoder, LoRA chỉ gắn vào phần ngôn ngữ. Vision encoder đã đọc chữ
    tốt sẵn; sửa nó bằng 3.500 ảnh là phá nhiều hơn được.
  · Mặt nạ nhãn: chỉ tính loss trên phần model phải sinh ra, không tính trên prompt.
    Thiếu bước này thì model học thuộc cả prompt và loss trông đẹp một cách giả tạo.

Chạy:
    ./venv_train/bin/python src/finetune_student.py --smoke     # 20 mẫu, kiểm đường ống
    ./venv_train/bin/python src/finetune_student.py             # thật
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", "/root/TANAHI/.cache/huggingface")
os.environ.setdefault("HOME", "/root/TANAHI/.home")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TRAIN_JSONL = ROOT / "data" / "processed" / "train.jsonl"
VAL_JSONL = ROOT / "data" / "processed" / "val.jsonl"
OUT_DIR = ROOT / "models" / "qwen25vl3b-lora"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Ảnh 512px → ~336 token thị giác với Qwen2.5-VL (ô 28×28). Đủ đọc chữ mà không nổ
# bộ nhớ khi huấn luyện trong slice MIG 20 GB.
# LẤY TỪ pipeline.py — KHÔNG định nghĩa lại. Train và serve phải resize ảnh y hệt nhau;
# lệch một chút là model học trên phân bố khác phân bố nó bị chấm.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import MAX_PIXELS, MIN_PIXELS, SYSTEM, USER_TMPL  # noqa: E402

# SYSTEM và USER_TMPL cũng LẤY TỪ pipeline.py, không định nghĩa lại.
# Trước đây file này giữ bản sao riêng của hai prompt đó. Nội dung khi kiểm (30/07 12:00)
# vẫn trùng, nhưng bản sao là mìn: prompt đã đổi 6 lần trong đêm 30/07, và sửa một bên mà
# quên bên kia thì model học trên prompt A rồi bị chấm trên prompt B — hỏng âm thầm, không
# lỗi nào nổ ra, chỉ có chất lượng tụt mà không ai biết vì sao.
# Đây là lần thứ năm cùng một loại lỗi trong dự án: hai chỗ định nghĩa cùng một thứ.


def build_target(rec: dict) -> str:
    """ocr_text ĐỨNG TRƯỚC brand — có chủ đích, không phải tuỳ ý sắp.

    Model sinh tuần tự, nên bắt nó PHIÊN ÂM chữ trong ảnh trước rồi mới khai brand là một
    ràng buộc mạnh: nó không thể nói "Clinique" nếu không viết được chữ đó ra ở bước trước.
    Và phần phiên âm này là nhân chứng thứ hai cho cổng neo lúc suy luận — cần thiết vì
    PaddleOCR bỏ sót 10,7% chữ logo (đo 30/07 trên 1.003 brand: chỗ ghi CLINIQUE nó trả
    về 'ORDY mallusion STRING SUBSILLE Momoids').
    """
    return json.dumps({"ocr_text": rec.get("ocr_text_model", "") or "",
                       "brand": rec.get("brand", "") or "",
                       "product": rec.get("product", "") or "",
                       "summary": rec.get("summary", "") or ""},
                      ensure_ascii=False)


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        sys.exit(f"Chưa có {p} — chạy src/build_dataset.py trước.")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--smoke", action="store_true", help="20 mẫu, 4 bước — kiểm đường ống")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiep", action="store_true",
                    help="chay tiep tu checkpoint gan nhat trong --out thay vi tu dau")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = load_jsonl(TRAIN_JSONL)
    val = load_jsonl(VAL_JSONL)
    if args.smoke:
        train, val = train[:20], val[:6]
        args.epochs = 1.0
    random.shuffle(train)

    print("=== Huấn luyện student Qwen2.5-VL-3B (LoRA) ===")
    print(f"  train / val : {len(train)} / {len(val)}")
    print(f"  epochs      : {args.epochs}   lr {args.lr}   LoRA r={args.lora_r}")
    print(f"  max_len     : {args.max_len}   ảnh ≤ {int(MAX_PIXELS**0.5)}px", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS,
                                         max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    model.config.use_cache = False

    # ĐÓNG BĂNG vision encoder — chỉ dạy phần ngôn ngữ cách viết summary đúng luật đề.
    n_frozen = 0
    for name, p in model.named_parameters():
        if "visual" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"  đóng băng vision: {n_frozen/1e6:.0f}M tham số")

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # Không gắn vào "visual" — exclude bằng cách chỉ định lớp ngôn ngữ
        exclude_modules=r".*visual.*",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    def encode(rec: dict):
        """Dựng một mẫu. Trả None nếu ảnh hỏng hoặc quá dài — bỏ qua, không làm chết run."""
        path = ROOT / rec["image"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:                                  # noqa: BLE001
            return None
        target = build_target(rec)
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": USER_TMPL.format(
                    ocr=(rec.get("ocr_text") or "(không đọc được chữ nào)")[:900])},
            ]},
        ]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + target + "<|im_end|>"
        batch = proc(text=[full], images=[img], return_tensors="pt")
        ids = batch["input_ids"][0]
        if ids.shape[0] > args.max_len:
            return None
        # MẶT NẠ: chỉ tính loss trên phần đáp án, phần prompt đặt -100.
        n_prompt = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = ids.clone()
        labels[:n_prompt] = -100
        batch["labels"] = labels.unsqueeze(0)
        return batch

    def run_batch(rec, train_mode: bool):
        b = encode(rec)
        if b is None:
            return None
        b = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in b.items()}
        if train_mode:
            out = model(**b)
        else:
            with torch.no_grad():
                out = model(**b)
        return out.loss

    def evaluate() -> float:
        model.eval()
        tot, n = 0.0, 0
        for rec in val:
            l = run_batch(rec, False)
            if l is not None and torch.isfinite(l):
                tot += l.item()
                n += 1
        model.train()
        return tot / n if n else float("nan")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = max(1, int(len(train) * args.epochs / args.grad_accum))
    warmup = max(1, int(0.03 * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    print(f"  bước tối ưu : {total_steps} (warmup {warmup})\n", flush=True)
    model.train()
    t0 = time.time()
    step = 0
    micro = 0
    running = 0.0
    n_skip = 0
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"

    n_micro_total = int(len(train) * args.epochs)
    order = [train[i % len(train)] for i in range(n_micro_total)]

    # ── CHAY TIEP SAU KHI BI NGAT ─────────────────────────────────────────────
    # `order` sinh ra tu `train` da shuffle voi seed co dinh, nen chay lai cung seed
    # ra dung thu tu cu. Chi can nho DA DI TOI CHI SO NAO la noi lai chinh xac.
    # Phai nap ca trang thai AdamW: khong co no thi momentum ve 0, loss nay len roi
    # phai hoc lai — coi nhu mat mot phan cong suc dinh cuu.
    bat_dau = 0
    meta_p = args.out / "train_meta.json"
    if args.tiep and meta_p.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        if m.get("trang_thai") == "xong":
            print("  ⚠️  checkpoint da o trang thai 'xong' — khong chay tiep nua.")
            return
        w = args.out / "adapter_model.safetensors"
        if not w.exists():
            print(f"  ⚠️  co train_meta nhung thieu {w.name} — chay lai tu dau.")
        else:
            set_peft_model_state_dict(model, load_file(str(w)))
            step = int(m.get("steps_done", 0))
            micro = int(m.get("micro_done", step * args.grad_accum))
            n_skip = int(m.get("n_skip", 0))
            if "i_order" in m:
                bat_dau = int(m["i_order"])
            else:
                # Checkpoint sinh boi ban CU (truoc 31/07 13:4x) khong ghi i_order.
                # Uoc luong: moi buoc toi uu = grad_accum mau khong bi bo, cong so bi bo.
                # Sai lech vai mau la chap nhan duoc — hon han train lai tu dau.
                bat_dau = step * args.grad_accum + n_skip
                print(f"  ⚠️  checkpoint bản CŨ, không có i_order — ước lượng {bat_dau}")
            op = args.out / "opt_state.pt"
            if op.exists():
                opt.load_state_dict(torch.load(op, map_location="cuda:0"))
                print(f"  ↳ đã nạp trạng thái AdamW")
            else:
                print("  ⚠️  thiếu opt_state.pt — momentum về 0, loss sẽ nảy một nhịp")
            print(f"  ↳ CHẠY TIẾP từ bước {step}/{total_steps} "
                  f"(mẫu {bat_dau}/{n_micro_total}, {bat_dau/n_micro_total*100:.1f}%)",
                  flush=True)

    for i_order in range(bat_dau, len(order)):
        rec = order[i_order]
        loss = run_batch(rec, True)
        if loss is None or not torch.isfinite(loss):
            n_skip += 1
            continue
        (loss / args.grad_accum).backward()
        running += loss.item()
        micro += 1
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            # LƯU ĐỊNH KỲ — bắt buộc, không phải tuỳ chọn.
            # Chỉ lưu ở cuối là đánh cược cả 2 giờ huấn luyện vào việc KHÔNG có gì ngắt
            # giữa chừng. Máy này là máy chia sẻ, và tiến trình của ta ĐÃ TỪNG bị giết từ
            # bên ngoài lúc 18:14 khi đang chạy khoẻ. Mất 2 giờ vì tiết kiệm 2 giây lưu là
            # đánh đổi tồi.
            if step % 100 == 0:
                model.save_pretrained(args.out)
                # Trang thai AdamW — thieu no thi `--tiep` mat momentum, loss nay len.
                torch.save(opt.state_dict(), args.out / "opt_state.pt")
                with open(args.out / "train_meta.json", "w", encoding="utf-8") as fh:
                    json.dump({"base_model": MODEL_ID, "steps_done": step,
                               "total_steps": total_steps, "loss": running / max(1, micro),
                               "trang_thai": "dang-chay",
                               # Ba so duoi day la thu cho phep `--tiep` noi lai DUNG CHO.
                               "i_order": i_order + 1, "micro_done": micro,
                               "n_skip": n_skip, "seed": args.seed,
                               "minutes": (time.time() - t0) / 60},
                              fh, ensure_ascii=False, indent=2)
                print(f"  ↳ đã lưu checkpoint ở bước {step} (mẫu {i_order+1})", flush=True)
            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                avg = running / max(1, micro)
                eta = el / step * (total_steps - step)
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  bước {step}/{total_steps}  loss {avg:.4f}  "
                      f"lr {lr_at(step):.2e}  {el/60:.1f}p  còn ~{eta/60:.0f}p  "
                      f"VRAM {mem:.1f}G  bỏ {n_skip}", flush=True)
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": step, "loss": avg,
                                         "lr": lr_at(step), "sec": el}) + "\n")
                running, micro = 0.0, 0
        if args.smoke and step >= 4:
            break

    vl = evaluate()
    print(f"\n  val loss: {vl:.4f}")
    model.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    with open(args.out / "train_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"base_model": MODEL_ID, "trang_thai": "xong",
                   "val_loss": vl, "steps": step,
                   "train_n": len(train), "val_n": len(val), "lora_r": args.lora_r,
                   "epochs": args.epochs, "lr": args.lr, "skipped": n_skip,
                   "minutes": (time.time() - t0) / 60}, fh, ensure_ascii=False, indent=2)
    print(f"=== Xong sau {(time.time()-t0)/60:.1f} phút → {args.out} ===")


if __name__ == "__main__":
    main()
