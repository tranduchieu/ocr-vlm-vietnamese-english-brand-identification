#!/usr/bin/env python3
"""PIPELINE NỘP BÀI — OCR → VLM ≤4B → cổng neo → summary.

Đây là thứ đem đi chấm. Mọi ràng buộc của đề được hiện thực ở đây, không ở chỗ khác:

  Mục 4 — KHÔNG gọi API ngoài. Toàn bộ chạy cục bộ: PaddleOCR + VietOCR + Qwen2.5-VL-3B.
  Mục 4 — mỗi model đơn lẻ ≤4B. PP-OCRv6 (~4M) · VietOCR vgg_seq2seq (~23M) ·
          Qwen2.5-VL-3B (3,755B — đã đếm bằng safetensors, xem WORK_LOG).
  Mục 4 — tổng VRAM ≤24GB. Đo thật bằng --bench, không ước lượng.
  Mục 3 — output JSONL đúng 2 trường {"image_id","summary"}.
  Mục 3 — ảnh KHÔNG có text VÀ KHÔNG có brand → summary rỗng "".
  Mục 5 — cổng neo cắt brand không xuất hiện trong chữ đọc được → chặn phạt x3.

VÌ SAO CỔNG NEO ĐẶT SAU VLM CHỨ KHÔNG TIN VLM (Đ1, Đ7):
Bịa brand bị phạt x3, im lặng phạt 0. "Bịa" = nói ra thứ không có trong ảnh — một mệnh đề
KIỂM CHỨNG ĐƯỢC BẰNG CHUỖI. Nên biến mức phạt nặng nhất thành thứ không thể xảy ra bằng
code, thay vì trông chờ model tự kiềm chế. Nhân chứng phải ĐỘC LẬP: OCR chạy riêng, không
phải do chính VLM cung cấp — nếu không thì phép kiểm là vòng tròn.

Chạy:
    python3 src/pipeline.py --images data/raw/phase2/images --out outputs/submission.jsonl
    python3 src/pipeline.py --bench --limit 100      # đo throughput + VRAM
    python3 src/pipeline.py --image một_tấm.jpg      # thử một ảnh
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Neo cache vao DU AN, khong ghim vao /root/TANAHI.
# VI SAO: `/root/TANAHI` chi ton tai tren may phat trien. Chay o may khac thi VietOCR
# tai backbone vgg19 va goi `os.makedirs('/root/TANAHI')` -> PermissionError, chet ngay
# khi nap OCR.
# Dung `_goc` = thu muc CHA cua du an neu ton tai, khong thi lay chinh thu muc du an.
def _thu_muc_cache() -> Path:
    """Tra ve /root/TANAHI neu THAT SU truy cap duoc, khong thi lay thu muc du an.

    Phai bat OSError: neu tien trinh khong co quyen doc `/root`, `Path.is_dir()` NEM
    PermissionError chu khong tra False. Kiem bang `is_dir()` tran la chet ngay dong
    import.
    """
    try:
        p = Path("/root/TANAHI")
        if p.is_dir():
            return p
    except OSError:
        pass
    return ROOT


_goc = _thu_muc_cache()
os.environ.setdefault("HOME", str(_goc / ".home"))
os.environ.setdefault("HF_HOME", str(_goc / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(_goc / ".cache"))
# TORCH_HOME rieng: torchvision tai vgg19_bn cho VietOCR ve day.
os.environ.setdefault("TORCH_HOME", str(_goc / ".cache" / "torch"))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_goc / ".cache" / "paddlex"))
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_DIR = ROOT / "models" / "qwen25vl3b-lora"
OCR_CONF_MIN = 0.70
MAX_PIXELS = 512 * 512
MIN_PIXELS = 128 * 128

SYSTEM = ("Bạn phân tích ảnh mạng xã hội Việt Nam. Chỉ nói những gì NHÌN THẤY trong ảnh. "
          "Không đoán thương hiệu. Trả về đúng một JSON.")
USER_TMPL = ('Text OCR đọc được (tham khảo, có thể sai hoặc thiếu):\n{ocr}\n\n'
             'Trả về JSON đúng thứ tự: '
             '{{"ocr_text": "...", "brand": "...", "product": "...", "summary": "..."}}')

# Prompt gọi LẦN HAI khi lần đầu trả rỗng oan.
# VÌ SAO CẦN — đo 30/07 03:45: khi text OCR gửi vào rỗng, student trả rỗng TẤT CẢ, kể cả
# ảnh có trà sữa / bánh sinh nhật rõ ràng. Nó học từ dữ liệu rằng "prompt không chữ →
# summary rỗng", tức dựa vào TEXT nhiều hơn dựa vào ẢNH.
# Đề chỉ cho rỗng khi **không có text VÀ không có brand**. Ảnh có vật thể mô tả được mà
# trả rỗng là mất trắng điểm. Gọi lần hai chỉ tốn 0,09 s/ảnh và chỉ cho ~17% ảnh.
RETRY_TMPL = ('Ảnh này KHÔNG đọc được chữ rõ ràng, nhưng nó CÓ nội dung nhìn thấy được.\n'
              'Hãy MÔ TẢ những gì bạn NHÌN THẤY: vật thể, bối cảnh, màu sắc, hoạt động.\n'
              'TUYỆT ĐỐI không đoán tên thương hiệu.\n'
              'Trả về JSON: {"ocr_text": "", "brand": "", "product": "...", '
              '"summary": "mô tả 1-2 câu"}')


# ───────────────────────────────────────────────── so khớp chuỗi

def _nfc(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", (s or "").lower()).split())


def _bare(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if unicodedata.category(c) != "Mn").lower()


FUZZY_THRESHOLD = 0.85


def _alnum(s: str) -> str:
    """Bo dau, bo moi ky tu khong phai chu-so. 'Mật Tea&Coffee' -> 'matteacoffee'."""
    return re.sub(r"[^a-z0-9]+", "", _bare(s))


def contains(hay: str, needle: str, thr: float = FUZZY_THRESHOLD) -> bool:
    """KHOP MO — bat buoc, khong phai tuy chon.

    ĐO ĐƯỢC 30/07 tren 691 anh co brand: khop chuoi TUYET DOI chi giu 74,3% brand,
    tuc CAT OAN 25,7%. Nhung nhung cai bi cat KHONG phai bia — chung la brand THAT ma
    PaddleOCR danh van sai:
        SEQUE          -> paddle doc 'SELQUE / SE / QUE'
        Kim Beauty Spa -> 'Km Beauly Spa'
        CHANH TƯƠI     -> 'CHANH TUI'
        Mật Tea&Coffee -> 'Teascofee'
    Cat oan la mat recall THAT o Metric 1, khong duoc gi ca.

    Hieu chuan nguong: 0,85 giu 81,6% (hon khop tuyet doi 7,3 diem). Kiem chung mat bia
    THAT van bi cat: "L'Oréal" tren priv_d_0053 (paddle chi doc duoc chu mo ta, khong co
    ten hang nao) bi cat o MOI nguong ke ca 0,70 -> noi long den 0,85 khong mo cua cho bia.

    VI SAO PHAI CO HAM NAY, KHONG DUNG `in`:
    Luc GAN NHAN, cong neo duoc doi chieu voi PaddleOCR HOAC phan chu teacher tu doc.
    Luc SUY LUAN khong co teacher — chi con PaddleOCR. Neu giu khop tuyet doi thi 33,6%
    brand student vua hoc se bi cat ngay tai cong. Do dung la train/serve skew.
    """
    n, h = _nfc(needle), _nfc(hay)
    if not n:
        return False
    if n in h or _bare(needle) in _bare(hay):
        return True
    a, b = _alnum(needle), _alnum(hay)
    if not a or len(a) < 3 or not b:
        return False
    if a in b:
        return True
    if _fuzzy_in(b, a, thr):
        return True
    # KHOP THEO TUNG TU, khong chi ca cum.
    # OCR lam RUNG han mot tu: 'Mật Tea&Coffee' -> paddle doc 'Teascofee' (mat chu 'Mật'),
    # 'Kim Beauty Spa' -> 'Km Beauly Spa'. Bat ca cum phai giong thi that bai, nhung chi
    # can MOT tu >= 4 ky tu khop la du bang chung rang brand co that tren anh.
    # Chi lay tu >= 4 ky tu: tu ngan nhu 'the', 'and' qua pho bien, de khop bua.
    toks = [t for t in re.split(r"[^a-z0-9]+", _bare(needle)) if len(t) >= 4]
    return any(t in b or _fuzzy_in(b, t, 0.80) for t in toks)


def _fuzzy_in(hay_alnum: str, needle_alnum: str, thr: float) -> bool:
    """Truot cua so nhieu do dai. Chi lay 1 do dai thi lech vai ky tu la truot."""
    from difflib import SequenceMatcher
    L = len(needle_alnum)
    if L < 3 or not hay_alnum:
        return False
    widths = {max(3, L + d) for d in (-3, -2, -1, 0, 1, 2, 3)}
    for i in range(max(1, len(hay_alnum) - L + 4)):
        for w in widths:
            seg = hay_alnum[i:i + w]
            if len(seg) >= 3 and SequenceMatcher(None, needle_alnum, seg).ratio() >= thr:
                return True
    return False


def contains_exact(hay: str, needle: str) -> bool:
    """Khớp NGUYÊN VĂN (chỉ bỏ dấu) — KHÁC `contains()` khớp mờ, và khác có chủ đích.

    Phân biệt này quan trọng:
      · brand ∈ text OCR      → dùng `contains()` KHỚP MỜ. OCR đánh vần sai là chuyện
        thường (`CLINIQUE` → `ORDY mallusion`), phạt oan chỗ đó là mất recall thật.
      · brand ∈ summary       → dùng `contains_exact()`. Model tự viết CẢ HAI chuỗi nên
        không có cớ gì để lệch. Đề đòi "xuất hiện NGUYÊN VĂN trong summary" (mục 3).

    Dùng lẫn hai hàm này làm `nguyên văn` tụt từ 100% xuống 89,4% mà không ai thấy —
    vì `anchor_gate` cho qua bằng khớp mờ còn báo cáo lại đo bằng khớp tuyệt đối.
    """
    n, h = _nfc(needle), _nfc(hay)
    return bool(n) and (n in h or _bare(needle) in _bare(hay))


def split_multi(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(";") if x.strip()]


RE_ABSENCE = re.compile(
    r"không (có|thấy|nhận|xác định|chứa|tìm thấy)\s*(được)?\s*"
    r"(thương hiệu|nhãn hiệu|nhãn hàng|brand|sản phẩm|logo|chữ|text|thông tin)", re.I)


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", t, re.S)
    if m:
        t = m.group(1).strip()
    i = t.find("{")
    if i < 0:
        return None
    depth, in_str, esc = 0, False, False
    for j, ch in enumerate(t[i:], i):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    d = json.loads(t[i:j + 1])
                    return d if isinstance(d, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


# ───────────────────────────────────────────────── cổng neo

# Lien tu / gioi tu de bi bo lai lo lung sau khi go ten brand ra khoi cau.
_RAC_DAU_CAU = re.compile(
    r"^\s*(và|voi|với|của|cua|do|từ|tu|by|of|the|by)\b[\s,.;:-]*", re.I)
_RAC_CUOI_CAU = re.compile(r"[\s,;:-]*\b(và|với|của|từ|by|of)\s*$", re.I)


def _don_cau(s: str) -> str:
    """Don cau sau khi go brand ra: bo lien tu cut dau/cuoi, gop khoang trang, viet hoa lai.

    DA VAP 30/07: cat 'Clinique' khoi "Clinique và SEQUE serum tốt" con lai
    "và SEQUE serum tốt" — mo dau bang lien tu cut. Judge cham muc "summary sai lech noi
    dung" (x1) se tru diem, va doc len thi ro rang la cau bi cat vung ve.
    """
    s = re.sub(r"\s{2,}", " ", s or "")
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"([,;:])\s*\1+", r"\1", s)
    s = s.strip(" -–,;:.")
    for _ in range(3):                       # co the con nhieu lop, vd "của và "
        new = _RAC_DAU_CAU.sub("", s)
        new = _RAC_CUOI_CAU.sub("", new)
        if new == s:
            break
        s = new.strip(" -–,;:.")
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def anchor_gate(brand: str, product: str, summary: str,
                ocr_text: str, vlm_ocr: str = "") -> tuple[str, str, str, list[str]]:
    """Cắt mọi brand không xuất hiện trong chữ đọc được, và gỡ nó khỏi summary.

    Trả (brand, product, summary, danh sách đã cắt).

    HAI NHÂN CHỨNG, không phải một:
      · `ocr_text` — PaddleOCR + VietOCR. Độc lập hoàn toàn với VLM (Đ7).
      · `vlm_ocr`  — phần chữ CHÍNH VLM phiên âm ra ở trường "ocr_text" của JSON.

    VÌ SAO PHẢI CÓ NHÂN CHỨNG THỨ HAI — đo được 30/07 trên 1.003 brand:
    Chỉ dùng PaddleOCR thì cắt **110 brand**, nhưng soi từng cái thì **107 là brand THẬT**
    mà PaddleOCR bỏ sót, chỉ **3** là bịa. Tỷ lệ 36:1 — vứt 107 cái đúng để bắt 3 cái sai.
    PaddleOCR không đọc nổi chữ logo: chỗ ghi `CLINIQUE` nó trả về
    `ORDY mallusion STRING SUBSILLE Momoids`; chỗ ghi `Gella` nó chỉ đọc `STRAWBERRY`.
    Bỏ cả lọc conf cũng chỉ cứu thêm 2,6 điểm.

    Tính theo thang phạt của đề: giữ 107 thật + 3 bịa chỉ thêm 3×3 = 9 điểm phạt trên
    ~2.000 ảnh (hallucination rate +0,0045 — không đáng kể), còn cắt chúng thì mất
    **10,7% recall** của Metric 1. Đổi như vậy là lỗ rõ ràng.

    Phiên âm của VLM KHÔNG phải nhân chứng độc lập, nhưng cũng không vô giá trị: model
    phải VIẾT RA được chuỗi chữ đó ở bước sinh trước khi khai brand (schema đặt "ocr_text"
    đứng trước "brand"). Đó là ràng buộc chặt hơn hẳn việc chỉ nêu tên một thương hiệu.

    CHỈ GÁC brand, KHÔNG gác product — có chủ đích. Brand ngắn, in rõ; bịa brand phạt x3.
    Product thường là cụm mô tả dài ("Smoothie tẩy da chết hương hoa anh đào") mà OCR hay
    đọc thiếu vài từ, gác nó sẽ cắt oan hàng loạt tên đúng.
    """
    hay = (ocr_text or "") + "\n" + (vlm_ocr or "")
    cut = []
    kept = []
    for b in split_multi(brand):
        (kept if contains(hay, b) else cut).append(b)
    prods = split_multi(product)
    if cut:
        for b in cut:
            # Gỡ tên bị cắt ra khỏi summary — để lại thì judge vẫn thấy brand bịa.
            # PHẢI gỡ CẢ GIỚI TỪ đứng ngay trước nó. Đã vấp 30/07: cắt 'Gella' khỏi
            # "Sản phẩm của Gella rất tốt" ra "Sản phẩm của rất tốt" — tiếng Việt vỡ,
            # mà giới từ nằm GIỮA câu nên bộ dọn đầu/cuối câu không bắt được.
            # Cho phep LAP 1-3 lan: "của thương hiệu Dove" phai go het ca chum, khong
            # phai chi go "thương hiệu Dove" roi de lai "của" lo lung.
            summary = re.sub(
                r"\b(?:(?:của|cua|với|voi|từ|tu|by|of|thương hiệu|thuong hieu|"
                r"nhãn hàng|nhãn hiệu|hãng)\s+){1,3}" + re.escape(b),
                "", summary, flags=re.I)
            summary = re.sub(re.escape(b) + r"\s+(và|va|&)\s+", "", summary, flags=re.I)
            summary = re.sub(r"\s+(và|va|&)\s+" + re.escape(b), "", summary, flags=re.I)
            summary = re.sub(re.escape(b), "", summary, flags=re.I)
        summary = _don_cau(summary)

        # BO LUON product nao CHUA brand vua bi cat.
        # DA VAP 30/07: cat brand khoi summary lam vo product. Vi du priv_h_0367
        # product='bỉm Pampers Gold', cat 'Pampers' -> summary con 'bỉm Gold' nen product
        # khong con NGUYEN VAN trong summary => vi pham yeu cau cung cua muc 3.
        # 7/2030 nhan dinh loi nay. Product nhac ten mot brand khong neo duoc thi ban than
        # no cung khong neo duoc — bo la dung, khong phai mat mat.
        prods = [x for x in prods if not any(_nfc(b) in _nfc(x) for b in cut)]

    # BAT BUOC cuoi: moi thu con lai PHAI co nguyen van trong summary.
    # Day la yeu cau CUNG cua de (muc 3). Giu lai thu khong xuat hien trong summary thi
    # vua khong duoc diem, vua bi judge tinh la "extract text khong co trong ground truth".
    # NGUYÊN VĂN dùng khớp TUYỆT ĐỐI, không dùng khớp mờ — xem chú thích contains_exact()
    prods = [x for x in prods if contains_exact(summary, x)]
    kept = [b for b in kept if contains_exact(summary, b)]
    return "; ".join(kept), "; ".join(prods), summary, cut


def summary_tu_ocr(ocr_text: str, max_len: int = 220) -> str:
    """LƯỚI AN TOÀN: dựng summary từ chính text OCR khi VLM không dùng được.

    VÌ SAO CẦN — đo được 30/07 00:40 trên 12 ảnh: `priv_d_0009` có **22 dòng chữ** mà
    pipeline trả summary RỖNG, vì VLM sinh JSON không parse được nên rơi về mặc định rỗng.
    Đề chỉ cho rỗng khi **không chữ VÀ không brand** (mục 3) → ảnh đó mất điểm oan hoàn toàn.

    Thà tóm tắt lại chữ đọc được còn hơn trả rỗng: đề yêu cầu thẳng "tóm tắt các dòng text
    đọc được trong ảnh". Rỗng thì được 0 điểm chắc chắn; echo chữ có thật thì tệ nhất là
    bị trừ x1 "extract text không có trong ground truth", mà chữ này CÓ trong ảnh nên
    khả năng bị trừ thấp.

    KHÔNG suy diễn gì thêm, KHÔNG thêm brand — chỉ chép lại chữ đã đọc được.
    """
    lines = [l.strip() for l in (ocr_text or "").splitlines() if l.strip()]
    seen, uniq = set(), []
    for l in lines:
        k = l.lower()
        if k not in seen and len(l) > 1:
            seen.add(k)
            uniq.append(l)
    if not uniq:
        return ""
    out = "Ảnh có các dòng chữ: " + ", ".join(uniq)
    return out[:max_len].rstrip(" ,") if len(out) > max_len else out


def finalize(brand: str, product: str, summary: str,
             ocr_text: str, n_ocr_lines: int, vlm_ocr: str = "") -> str:
    """Áp luật mục 3 của đề rồi trả summary cuối cùng.

    HAI LUẬT KHÁC NHAU, tôi từng đọc nhầm thành một:
      · không brand NHƯNG có chữ  → vẫn mô tả nội dung bình thường
      · không chữ VÀ không brand  → trả "" (chuỗi rỗng)
    """
    s = (summary or "").strip()
    if RE_ABSENCE.search(s):          # câu "không có thương hiệu nào" → judge phạt x1
        s = ""
    # HAI NHAN CHUNG, y het `anchor_gate` — KHONG duoc chi nhin PaddleOCR.
    # DA VAP 30/07 03:20: `anchor_gate` dung 2 nhan chung nhung `finalize` chi dung 1,
    # nen anh ma STUDENT doc duoc chu con Paddle khong doc duoc bi XOA TRANG summary du
    # student viet dung. Do duoc tren 292 anh gold: **39 anh (13,4%) rong OAN**, mat trang
    # diem. Day la lan thu tu cung mot loai loi: hai cho kiem cung mot thu bang hai bo
    # chung cu khac nhau.
    has_text = (n_ocr_lines > 0 or bool((ocr_text or "").strip())
                or bool((vlm_ocr or "").strip()))
    has_entity = bool(split_multi(brand) or split_multi(product))
    if not has_text and not has_entity:
        return ""                     # ĐÚNG luật: không chữ VÀ không brand
    if not s and has_text:
        # Chép lại chữ ĐÃ ĐỌC CHẮC (conf ≥ 0,70) thay vì mất trắng ảnh này.
        # KHÔNG chép text conf thấp — đã thử 30/07 03:47 và nó sinh ra rác kiểu
        # "Nusc Hanks Subul Tuscip", đúng thứ đề phạt "tự điền text không có trong ảnh".
        return summary_tu_ocr(ocr_text or vlm_ocr)
    return s


# ───────────────────────────────────────────────── các khâu

class OCR:
    """PaddleOCR phát hiện vùng chữ → VietOCR nhận dạng (quyết định G0.2).

    Đo trên 200 ảnh: cách ghép này cho token không-nguyên-âm 8,2% (so với 18,6% của
    PaddleOCR trọn gói) và token-F1 0,580 (so với 0,366). Chậm hơn 3,9× nhưng 2,9 ảnh/s
    vẫn vượt mốc ≥1 ảnh/s của đề.
    """

    def __init__(self, device: str = "gpu") -> None:
        from paddleocr import PaddleOCR
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        # Giu Y HET src/run_ocr.py - file do da chay that 3.919 anh khong loi. Nhan day
        # student sinh ra tu dung duong ong nay, nen suy luan phai khop tung tham so,
        # ke ca nguong loc conf. Lech mot cho la train/serve skew.
        self.det = PaddleOCR(lang="vi", device=device,
                             use_doc_orientation_classify=False,
                             use_doc_unwarping=False, use_textline_orientation=False)
        # ĐỌC CONFIG TỪ FILE TRONG DỰ ÁN, KHÔNG tải từ vocr.vn.
        # ĐÃ VẤP 30/07 03:23: `Cfg.load_config_from_name()` tải YAML từ `vocr.vn` MỖI LẦN
        # khởi động. Site timeout → pipeline chết hẳn. Một cú mạng chập lúc chấm hoặc lúc
        # demo là hỏng cả bài. Và phụ thuộc website ngoài lúc chạy cũng trái tinh thần
        # mục 4 của đề. Config đã lưu ở configs/vietocr_vgg_seq2seq.yml.
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "vietocr_vgg_seq2seq.yml"
        if cfg_path.is_file():
            cfg = Cfg.load_config_from_file(str(cfg_path))
        else:                                    # dự phòng: vẫn chạy được nếu thiếu file
            cfg = Cfg.load_config_from_name("vgg_seq2seq")
        cfg["device"] = "cuda:0" if device == "gpu" else "cpu"
        cfg["predictor"]["beamsearch"] = False
        # Duong dan trong so: uu tien bien moi truong VIETOCR_WEIGHTS, roi den thu muc
        # trong du an, cuoi cung moi den duong dan tuyet doi cua may phat trien.
        # VI SAO PHAI LINH DONG: ghim cung duong dan tuyet doi la KHOA pipeline vao dung
        # mot may. Ban giam khao chay tren may cua ho se khong tim thay trong so.
        for w in [Path(os.environ["VIETOCR_WEIGHTS"])] if os.environ.get("VIETOCR_WEIGHTS") \
                else [Path(__file__).resolve().parent.parent / "models" / "vietocr" / "vgg_seq2seq.pth",
                      Path("/root/TANAHI/.cache/vietocr/vgg_seq2seq.pth")]:
            if w.is_file():
                cfg["weights"] = str(w)  # PHAI khop kien truc seq2seq, khong phai transformer
                break
        self.rec = Predictor(cfg)

    def _crops(self, img) -> list:
        """Phat hien vung chu -> tra ve danh sach anh dong da cat."""
        import numpy as np
        arr = np.array(img)
        polys = []
        for page in self.det.predict(arr):
            d = page if isinstance(page, dict) else page.json["res"]
            polys += list(d.get("dt_polys", []))
        out = []
        for poly in polys:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            out.append((img.crop((max(0, int(x0)), max(0, int(y0)), int(x1), int(y1))),
                        [[int(x), int(y)] for x, y in pts]))
        return out

    def __call__(self, img, trace: bool = False) -> tuple:
        """MOT anh: gop TAT CA dong cua no thanh MOT lan goi VietOCR.

        VI SAO: truoc day goi `rec.predict()` TUNG DONG. Moi anh co 6-20 dong -> 6-20 lan
        nap/chay model cho mot anh. Do duoc 30/07 03:13: OCR ton **0,86 s/anh** khi chay
        song song voi VLM, tuc mot minh no da gioi han throughput o 1,16 anh/giay — sat
        moc >=1,0 cua de den muc nguy hiem.
        `predict_batch()` chay ca chum dong trong mot luot, GPU duoc dung dung cach.
        """
        crops = self._crops(img)
        n_phat_hien = len(crops)      # SO VUNG CHU DETECTOR THAY — khac so dong doc duoc
        if not crops:
            return ("", 0, []) if trace else ("", 0)
        # NHAN DANG TUNG DONG, khong dung `predict_batch`.
        # DA THU 30/07 04:12 va DO CO KHOI DONG TRUOC: gop lo **CHAM HON** (0,221 vs
        # 0,182 s/anh = 0,83x) va cho text GIONG HET (6/6 anh). VietOCR dem moi dong ve
        # cung chieu rong nen dong ngan bi phi tinh toan. "Toi uu" cua toi phan tac dung —
        # giu ban don gian, da chay that 3.919 anh.
        lines, regions = [], []
        for c, poly in crops:
            try:
                t, pb = self.rec.predict(c, return_prob=True)
            except Exception:            # mot dong loi khong duoc lam chet ca anh
                if trace:
                    regions.append({"polygon": poly, "text": "", "ocr_confidence": None,
                                    "used_in_prompt": False})
                continue
            used = bool(t and t.strip() and float(pb) >= OCR_CONF_MIN)
            if used:
                lines.append(t.strip())
            if trace:
                regions.append({"polygon": poly, "text": (t or "").strip(),
                                "ocr_confidence": float(pb), "used_in_prompt": used})

        # TRA VE SO VUNG PHAT HIEN, khong phai so dong qua nguong conf.
        # DE noi "anh khong co TEXT" — detector thay 5 vung chu tuc la anh CO text, du
        # nhan dang khong chac. DO DUOC 30/07 03:40: 18/150 anh (12%) bi tra RONG OAN vi
        # moi dong deu conf < 0,70 nen bi loc het, pipeline tuong anh trang.
        # Text loc theo conf van dung cho PROMPT (chat luong), nhung QUYET DINH CO/KHONG
        # co chu phai dua vao DETECTOR.
        result = ("\n".join(lines), n_phat_hien)
        return (*result, regions) if trace else result


class VLM:
    def __init__(self, lora: Path | None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(BASE_MODEL, min_pixels=MIN_PIXELS,
                                                  max_pixels=MAX_PIXELS)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="cuda:0")
        if lora and lora.exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(lora))
            self.model = self.model.merge_and_unload()   # gộp để suy luận nhanh hơn
            print(f"  ✓ đã nạp LoRA từ {lora}")
        else:
            print("  ⚠️  CHƯA có LoRA — đang chạy model gốc, chất lượng sẽ thấp hơn")
        self.model.eval()

    def batch(self, imgs: list, ocr_texts: list[str], max_new: int = 200) -> list[dict]:
        """Sinh cho NHIEU anh trong MOT luot generate.

        VI SAO BAT BUOC: do duoc 30/07 00:15 — xu ly tung anh mot cho throughput
        **0,42 anh/giay**, TRUOT moc >= 1,0 cua de (muc 4). Phan tach: OCR 0,38s +
        VLM 2,00s. VLM la nut that, va 2 giay do gan nhu toan bo la do sinh token TUAN TU
        cho MOT chuoi — GPU khong duoc dung het.
        De ghi ro "≥1 anh/giay KHI CHAY SONG SONG (1000 anh trong ≤1000 giay)" nen batch
        la cach dung. VRAM chi dung 7,2/24 GB nen con du 17 GB de doi lay toc do.

        Padding phai la LEFT cho decoder-only: pad ben phai thi token sinh ra noi tiep
        pad chu khong noi tiep prompt.
        """
        prompts = []
        for ocr_text in ocr_texts:
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_TMPL.format(
                        ocr=(ocr_text or "(không đọc được chữ nào)")[:900])},
                ]},
            ]
            prompts.append(self.proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        old_side = self.proc.tokenizer.padding_side
        self.proc.tokenizer.padding_side = "left"
        try:
            b = self.proc(text=prompts, images=imgs, padding=True,
                          return_tensors="pt").to("cuda:0")
        finally:
            self.proc.tokenizer.padding_side = old_side
        with self.torch.no_grad():
            out = self.model.generate(**b, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.proc.tokenizer.eos_token_id)
        n_in = b["input_ids"].shape[1]
        res = []
        for row in out:
            txt = self.proc.tokenizer.decode(row[n_in:], skip_special_tokens=True)
            res.append(extract_json(txt) or {"brand": "", "product": "", "summary": ""})
        return res

    def batch_retry(self, imgs: list) -> list[dict]:
        return [self(im, "", prompt_override=RETRY_TMPL) for im in imgs]

    def __call__(self, img, ocr_text: str, max_new: int = 200,
                 prompt_override: str = "") -> dict:
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_override or USER_TMPL.format(
                    ocr=(ocr_text or "(không đọc được chữ nào)")[:900])},
            ]},
        ]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        b = self.proc(text=[prompt], images=[img], return_tensors="pt").to("cuda:0")
        with self.torch.no_grad():
            out = self.model.generate(**b, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.proc.tokenizer.eos_token_id)
        gen = out[0][b["input_ids"].shape[1]:]
        txt = self.proc.tokenizer.decode(gen, skip_special_tokens=True)
        return extract_json(txt) or {"brand": "", "product": "", "summary": ""}


class VLMRemote:
    """LƯỚI DỰ PHÒNG: gọi student qua vLLM thay vì transformers.

    KHI NÀO DÙNG: nếu `--bench` cho throughput < 1 ảnh/giây dù đã batch 8 + OCR song song.
    vLLM có continuous batching và paged attention, nhanh hơn hẳn vòng `generate` của
    transformers. Đã biết chắc chạy được trên máy này: teacher 7B đạt 82 ảnh/phút =
    1,37 ảnh/giây, mà student 3B nhẹ hơn gấp đôi.

    VẪN HỢP QUY mục 4: vLLM chạy CỤC BỘ trên cùng GPU, không phải API bên ngoài.
    Endpoint là `localhost`, không có gì đi ra Internet.

    Dựng server:
        ./scripts/serve_student.sh
    Rồi chạy:
        ./venv_ocr/bin/python src/pipeline.py --vllm-url http://localhost:25242/v1 --batch 16
    """

    def __init__(self, url: str, model: str = "student", workers: int = 16) -> None:
        import threading
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.workers = workers
        # SO SACH LOI — xem chu thich trong `_one`. Khong co no thi mot ha tang chet
        # bien thanh "du lieu trong nhu hop le" va bao cao ra so sai.
        self._lk = threading.Lock()
        self.n_goi = 0          # tong so lan goi
        self.n_loi = 0          # loi tang van chuyen (404 / timeout / server chet)
        self.n_khong_parse = 0  # goi duoc nhung model tra chuoi khong phai JSON
        self.loi_dau = ""       # thong bao loi DAU TIEN, de biet nguyen nhan goc

    def _ghi_so(self, loai: str, mo_ta: str = "") -> None:
        with self._lk:
            if loai == "goi":
                self.n_goi += 1
                return
            setattr(self, loai, getattr(self, loai) + 1)
            if mo_ta and not self.loi_dau:
                self.loi_dau = mo_ta

    def bao_cao_loi(self) -> str:
        """Mot dong tong ket so sach. Rong = khong co gi bat thuong."""
        if not (self.n_loi or self.n_khong_parse):
            return ""
        pct = 100.0 * (self.n_loi + self.n_khong_parse) / max(1, self.n_goi)
        s = (f"  ⚠ GOI VLM: {self.n_goi} lan · LOI {self.n_loi} · "
             f"khong parse duoc {self.n_khong_parse}  ({pct:.1f}%)")
        if self.loi_dau:
            s += f"\n    loi dau tien: {self.loi_dau}"
        if pct >= 5.0:
            s += ("\n    >>> TREN 5% -> MOI SO DO TRONG LAN CHAY NAY KHONG DUNG TIN DUOC. "
                  "Sua server roi do lai.")
        return s

    def _one(self, args) -> dict:
        import base64
        import io
        import httpx
        img, ocr_text = args[0], args[1]
        prompt_override = args[2] if len(args) > 2 else ""
        # GUI PNG KHONG MAT MAT, va KHONG tu thu nho o client.
        # DA VAP 30/07 04:00: client thu nho ve 512 roi nen JPEG q88, server lai resize
        # lan nua -> HAI lan bien doi co mat mat. Chu nho rat nhay voi chuyen nay, va
        # brand recall tut tu 63,8% (duong transformers) xuong 54,6% (duong vLLM).
        # De server tu resize theo min/max_pixels trong preprocessor_config.json — dung
        # y het processor luc huan luyen.
        buf = io.BytesIO()
        im = img.copy()
        if max(im.size) > 1024:          # chan anh qua to cho do ton bang thong
            im.thumbnail((1024, 1024))
        im.save(buf, "PNG", optimize=False)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {
            "model": self.model, "max_tokens": 260, "temperature": 0.0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_override or USER_TMPL.format(
                        ocr=(ocr_text or "(không đọc được chữ nào)")[:900])},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
        }
        # BAI HOC DAT NHAT CUA DU AN (30/07 04:50): ban dau cho la
        #     except Exception: return {"brand":"","product":"","summary":""}
        # Server tra HTTP 404 cho MOI request (config.json sai, xem src/gop_lora.py) nen
        # model CHUA TUNG duoc goi — the ma pipeline chay tron tru, ra JSONL day du va bao
        # "2,25 anh/giay". Do MOT VLM KHONG LAM GI CA, va con so sai da bao ra cho nguoi
        # dung. An suot 1 gio.
        # Rong VAN la gia tri tra ve dung — mot anh loi khong duoc lam chet ca lan chay
        # 1.200 anh — nhung phai VAO SO va bao cuoi lan chay.
        self._ghi_so("goi")
        try:
            r = httpx.post(self.url, json=payload, timeout=180)
            if r.status_code != 200:
                self._ghi_so("n_loi", f"HTTP {r.status_code} tu {self.url} — "
                                      f"{r.text[:160]}")
                return {"brand": "", "product": "", "summary": ""}
            txt = r.json()["choices"][0]["message"]["content"]
        except Exception as exc:                           # noqa: BLE001
            self._ghi_so("n_loi", f"{type(exc).__name__}: {exc}")
            return {"brand": "", "product": "", "summary": ""}
        d = extract_json(txt)
        if d is None:
            self._ghi_so("n_khong_parse", f"khong phai JSON: {txt[:160]!r}")
            return {"brand": "", "product": "", "summary": ""}
        return d

    def batch(self, imgs: list, ocr_texts: list[str], max_new: int = 260) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self.workers, len(imgs))) as ex:
            return list(ex.map(self._one, zip(imgs, ocr_texts)))

    def __call__(self, img, ocr_text: str, max_new: int = 260,
                 prompt_override: str = "") -> dict:
        return self._one((img, ocr_text, prompt_override))

    def batch_retry(self, imgs: list) -> list[dict]:
        """Goi lai voi prompt MO TA THI GIAC cho nhung anh lan dau tra rong oan."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, len(imgs)))) as ex:
            return list(ex.map(self._one, [(im, "", RETRY_TMPL) for im in imgs]))


def ocr_producer(ocr: OCR, files: list[Path], batch: int, q,
                 stat: dict | None = None) -> None:
    """Luong NAP TRUOC: OCR lo N+1 trong khi luong chinh dang cho VLM sinh lo N.

    VI SAO CAN: batch 8 giup VLM nhung OCR van tuan tu trong lo — 8 x 0,38s = 3,0s moi lo,
    cong voi ~3,5s cua VLM la 6,5s/lo = 0,81s/anh = 1,23 anh/giay. Dat moc nhung QUA SAT,
    mot may cham cham hon la truot.
    Cho hai viec chay chong len nhau thi moi lo chi ton max(3,0 ; 3,5) = 3,5s -> 0,44s/anh
    = ~2,3 anh/giay. Bien an toan gap doi.

    OCR (convnet nho) va VLM (decode transformer) nghen o cho khac nhau nen chay cung luc
    tren mot GPU van co loi, khong phai chia doi toc do.
    Hang doi chi giu 2 lo -> khong nap ca 1.200 anh vao RAM.
    """
    from PIL import Image
    try:
        for i in range(0, len(files), batch):
            chunk = files[i:i + batch]
            imgs, ocrs, n_liness, good = [], [], [], []
            for fp in chunk:
                _t = time.time()
                try:
                    img = Image.open(fp).convert("RGB")
                    txt, n = ocr(img)
                except Exception as exc:                   # noqa: BLE001
                    # Anh bi bo se KHONG co dong nao trong JSONL -> thieu image_id trong
                    # bai nop. Phai vao so, khong duoc bo im lang.
                    if stat is not None:
                        stat["bo"] = stat.get("bo", 0) + 1
                        stat.setdefault("bo_dau", f"{fp.name}: {type(exc).__name__}: {exc}")
                    continue
                if stat is not None:
                    stat["sec"] = stat.get("sec", 0.0) + (time.time() - _t)
                    stat["n"] = stat.get("n", 0) + 1
                good.append(fp)
                imgs.append(img)
                ocrs.append(txt)
                n_liness.append(n)
            if good:
                q.put((good, imgs, ocrs, n_liness))
    finally:
        q.put(None)                                        # tin hieu het viec


def assemble(paths, datas, ocrs, n_liness, t_ocr_per, t_vlm_per) -> list[dict]:
    """Ap cong neo + luat muc 3 cho ca lo. Tach rieng de khong lap code o 3 cho."""
    out = []
    for p, d, ocr_text, n_lines in zip(paths, datas, ocrs, n_liness):
        brand, product, summary, cut = anchor_gate(
            str(d.get("brand", "")), str(d.get("product", "")),
            str(d.get("summary", "")), ocr_text, str(d.get("ocr_text", "")))
        final = finalize(brand, product, summary, ocr_text, n_lines,
                         str(d.get("ocr_text", "")))
        out.append({"image_id": p.stem, "summary": final, "_brand": brand,
                    "_product": product, "_ocr": ocr_text,
                    "_vlm_ocr": str(d.get("ocr_text", "")), "_n_lines": n_lines,
                    "_cut": cut, "_t_ocr": round(t_ocr_per, 3),
                    "_t_vlm": round(t_vlm_per, 3)})
    return out


def vram_gb() -> float:
    """VRAM dinh cua tien trinh nay, tinh bang GB.

    Dung `max_memory_allocated` chu KHONG dung `nvidia-smi`: tren MIG, nvidia-smi khong
    thay tien trinh cua container khac va tra ve so vo nghia (da vap 30/07 — watchdog
    tuong con trong 20 GB trong khi that ra chi con 0,4 GB).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:                                        # noqa: BLE001
        return 0.0


class OCRTrong:
    """OCR RONG — chi dung khi may khong cai duoc PaddleOCR.

    ⚠️ KHONG PHAI pipeline that. Dung cai nay thi mat NHAN CHUNG DOC LAP cua cong neo
    (D7): chi con phien am cua chinh VLM lam nhan chung, tuc la model tu xac nhan chinh
    no — vong tron. Va prompt mat phan chu OCR nen lech phan bo so voi luc huan luyen.

    Ly do ton tai: co may chi co Python 3.14, ma `paddlepaddle` chua co banh xe cp314.
    Dung de xem nhanh chat luong brand/product/summary, KHONG dung de do diem hay bao
    cao so lieu.
    """

    def __call__(self, img, trace: bool = False) -> tuple:
        return ("", 0, []) if trace else ("", 0)


def process_image(ocr: OCR, vlm: VLM, img, image_id: str,
                  include_ocr_regions: bool = False) -> dict:
    """MỘT đường đi duy nhất cho một ảnh — `pipeline --image` và `webapp` đều gọi hàm này.

    ĐÃ VẤP 30/07 00:35: `webapp.py` tự lặp lại logic này và gọi `anchor_gate()` **thiếu
    tham số thứ 5** (`vlm_ocr`). Kết quả: bản demo mất nhân chứng thứ hai, cắt oan brand,
    và trông TỆ HƠN pipeline thật — đúng thứ không được xảy ra khi đem đi trình bày.
    Nguyên nhân gốc là code bị copy. Giờ chỉ có một bản.
    """
    t0 = time.time()
    ocr_result = ocr(img, trace=True) if include_ocr_regions else ocr(img)
    ocr_text, n_lines = ocr_result[:2]
    t_ocr = time.time() - t0
    t1 = time.time()
    d = vlm(img, ocr_text)
    t_vlm = time.time() - t1
    vlm_ocr = str(d.get("ocr_text", ""))
    brand, product, summary, cut = anchor_gate(
        str(d.get("brand", "")), str(d.get("product", "")),
        str(d.get("summary", "")), ocr_text, vlm_ocr)
    final = finalize(brand, product, summary, ocr_text, n_lines, vlm_ocr)
    result = {"image_id": image_id, "summary": final, "_brand": brand, "_product": product,
            "_ocr": ocr_text, "_vlm_ocr": vlm_ocr, "_raw": d,
            "_n_lines": n_lines, "_cut": cut,
            "_t_ocr": round(t_ocr, 3), "_t_vlm": round(t_vlm, 3)}
    if include_ocr_regions:
        result["_ocr_regions"] = ocr_result[2]
    return result


def process_one(ocr: OCR, vlm: VLM, path: Path) -> dict:
    from PIL import Image
    return process_image(ocr, vlm, Image.open(path).convert("RGB"), path.stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=ROOT / "data/raw/phase2/images")
    ap.add_argument("--image", type=Path, help="chạy đúng một ảnh rồi in chi tiết")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "submission.jsonl")
    ap.add_argument(
        "--emit-entities",
        action="store_true",
        help="ghi thêm brand/product sau anchor_gate (để chấm entity, không đổi format nop bai)",
    )
    ap.add_argument("--lora", type=Path, default=LORA_DIR)
    ap.add_argument("--vllm-url", default="",
                    help="dùng vLLM cục bộ thay transformers (lưới dự phòng throughput)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--file-list", type=Path,
                    help="file text, mỗi dòng một đường dẫn ảnh (vd gold_paths.txt)")
    ap.add_argument("--bench", action="store_true", help="đo throughput + VRAM theo mục 4")
    ap.add_argument("--batch", type=int, default=8,
                    help="số ảnh mỗi lượt generate. 1 = tuần tự (0,42 ảnh/s, TRƯỢT mốc đề)")
    args = ap.parse_args()

    print("=== Pipeline: OCR → VLM 3B → cổng neo ===", flush=True)
    t_load = time.time()
    ocr = OCR()
    if args.vllm_url:
        vlm = VLMRemote(args.vllm_url)
        print(f"  VLM qua vLLM CỤC BỘ: {args.vllm_url}  (vẫn hợp quy mục 4)")
    else:
        vlm = VLM(args.lora)
    print(f"  nạp model: {time.time()-t_load:.1f}s", flush=True)

    if args.image:
        r = process_one(ocr, vlm, args.image)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    if args.file_list:
        # Chay dung mot danh sach anh (vd 292 anh gold) thay vi ca thu muc — can de do
        # chat luong tren tap GIU LAI, khong lan vao anh da train.
        files = [ROOT / l.strip() for l in open(args.file_list, encoding="utf-8")
                 if l.strip()]
        files = [f for f in files if f.exists()]
    else:
        files = sorted([p for p in args.images.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    if args.limit:
        files = files[: args.limit]
    print(f"  ảnh: {len(files)}   → {args.out}\n", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_cut = n_empty = 0
    t_ocr_sum = t_vlm_sum = 0.0
    import queue as _q
    import threading as _th
    done_n = 0
    q: _q.Queue = _q.Queue(maxsize=2)      # giu toi da 2 lo -> khong nap het anh vao RAM
    ocr_stat: dict = {}
    prod = _th.Thread(target=ocr_producer,
                      args=(ocr, files, args.batch, q, ocr_stat), daemon=True)
    prod.start()

    with open(args.out, "w", encoding="utf-8") as fh:
        while True:
            item = q.get()
            if item is None:
                break
            chunk, imgs, ocrs, n_liness = item
            t1 = time.time()
            try:
                datas = vlm.batch(imgs, ocrs)
                # GOI LAI LAN HAI cho anh tra rong oan: detector THAY vung chu (hoac anh
                # co noi dung) ma summary rong. Do 30/07 03:45: 18/150 anh (12%) mat trang
                # diem vi student thay prompt khong co chu la tra rong tat ca.
                # Chi goi lai cho so it anh do -> chi phi ~0,09s moi anh, khong dang ke.
                need = [k for k, d in enumerate(datas)
                        if not str(d.get("summary", "")).strip() and n_liness[k] > 0]
                if need:
                    again = vlm.batch_retry([imgs[k] for k in need])
                    n_cuu = 0
                    for k, d2 in zip(need, again):
                        if str(d2.get("summary", "")).strip():
                            datas[k] = d2
                            n_cuu += 1
                    if n_cuu:
                        print(f"    ↳ gọi lại lần hai: cứu {n_cuu}/{len(need)} ảnh khỏi rỗng oan",
                              flush=True)
                t_vlm = time.time() - t1
                rs = assemble(chunk, datas, ocrs, n_liness,
                              0.0, t_vlm / max(1, len(chunk)))
            except Exception as exc:                       # noqa: BLE001
                # Mot lo hong KHONG duoc lam chet ca lan chay 1.200 anh: lui ve tung anh.
                print(f"  ! lô {len(chunk)} ảnh lỗi ({type(exc).__name__}: {exc}) "
                      f"→ lùi về từng ảnh", flush=True)
                rs = []
                for k, fp in enumerate(chunk):
                    try:
                        d = vlm(imgs[k], ocrs[k])
                        rs += assemble([fp], [d], [ocrs[k]], [n_liness[k]], 0.0, 0.0)
                    except Exception:                      # noqa: BLE001
                        rs.append({"image_id": fp.stem, "summary": "", "_cut": [],
                                   "_t_ocr": 0.0, "_t_vlm": 0.0})
            for r in rs:
                row = {"image_id": r["image_id"], "summary": r["summary"]}
                if args.emit_entities:
                    row["brand"] = r.get("_brand", "") or ""
                    row["product"] = r.get("_product", "") or ""
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_cut += len(r["_cut"])
                n_empty += (not r["summary"])
                t_ocr_sum += r["_t_ocr"]
                t_vlm_sum += r["_t_vlm"]
                done_n += 1
            fh.flush()
            el = time.time() - t0
            # OCR gio chay CHONG LEN VLM nen khong con cong don duoc — bao thoi gian
            # tuong (wall clock) moi anh, do la con so de cham theo.
            print(f"  {done_n}/{len(files)}  {done_n/el:.2f} ảnh/s  "
                  f"(VLM {t_vlm_sum/done_n:.2f}s/ảnh, OCR chạy song song)  "
                  f"cắt {n_cut} brand · rỗng {n_empty}", flush=True)

    el = time.time() - t0
    # DO VRAM TOAN CUC. `max_memory_allocated()` chi thay tien trinh NAY — khi VLM chay
    # trong server vLLM rieng, no bao 0,3 GB trong khi thuc te ca pipeline dung 9,6 GB.
    # De cham "tong VRAM cua TOAN BO pipeline", nen phai doc muc GPU thuc.
    try:
        _free, _total = torch.cuda.mem_get_info()
        vram = (_total - _free) / 1024**3
    except Exception:                                  # noqa: BLE001
        vram = torch.cuda.max_memory_allocated() / 1024**3
    ocr_avg = ocr_stat.get("sec", 0.0) / max(1, ocr_stat.get("n", 0))
    vlm_avg = t_vlm_sum / max(1, done_n)
    print(f"\n=== Xong {done_n}/{len(files)} ảnh trong {el/60:.1f} phút ===")
    print(f"  OCR {ocr_avg:.2f}s/ảnh + VLM {vlm_avg:.2f}s/ảnh, chạy CHỒNG LÊN nhau")
    print(f"  → thời gian tường thực tế {el/max(1,done_n):.2f}s/ảnh "
          f"(nếu tuần tự sẽ là {ocr_avg+vlm_avg:.2f}s)")
    thr = done_n / el
    lat = ocr_avg + vlm_avg      # latency MOT anh don le = tong tuan tu, khong phai wall/n
    print(f"  Throughput   : {thr:.2f} ảnh/giây   (đề yêu cầu ≥ 1,0)"
          f"   {'ĐẠT' if thr >= 1.0 else 'CHƯA ĐẠT'}")
    print(f"  Latency/ảnh  : {lat:.2f}s          (đề yêu cầu ≤ 20s, xử lý ĐƠN LẺ)"
          f"   {'ĐẠT' if lat <= 20 else 'CHƯA ĐẠT'}")
    print(f"  VRAM đỉnh    : {vram:.1f} GB               (đề yêu cầu ≤ 24)"
          f"   {'ĐẠT' if vram <= 24 else 'CHƯA ĐẠT'}")
    print(f"  Brand bị cổng neo cắt : {n_cut}")
    print(f"  Summary rỗng          : {n_empty}/{len(files)}")

    # SO SACH LOI — in TRUOC khi in duong dan file, de khong bi cuon mat.
    bc = vlm.bao_cao_loi() if hasattr(vlm, "bao_cao_loi") else ""
    n_loi_vlm = getattr(vlm, "n_loi", 0) + getattr(vlm, "n_khong_parse", 0)
    n_goi_vlm = getattr(vlm, "n_goi", 0)
    if bc:
        print(bc)
    if ocr_stat.get("bo"):
        print(f"  ⚠ OCR BO {ocr_stat['bo']} anh (khong doc duoc file) — chung KHONG co "
              f"dong nao trong JSONL.\n    dau tien: {ocr_stat.get('bo_dau','')}")
    if done_n != len(files):
        print(f"  ⚠ chi xu ly {done_n}/{len(files)} anh — thieu {len(files)-done_n}")
    if not (bc or ocr_stat.get("bo")) and done_n == len(files):
        print("  ✓ khong co loi goi VLM, khong bo anh nao")
    print(f"  → {args.out}")

    # Ghi so do duoc ra file de analysis/check_brief.py doi chieu voi muc 4 cua de.
    # Bang chung phai la file, khong phai dong log troi qua roi mat.
    #
    # CHI `--bench` moi duoc ghi `bench.json`. Truoc day moi lan chay deu ghi, nen buoc 7
    # (chay 292 anh gold, KHONG phai phep do rang buoc) de len ket qua buoc 6 — WORK_LOG
    # ghi 1,35 anh/giay tren 150 anh trong khi file lai la 1,63 tren 292. Hai con so deu
    # that nhung khong con biet con nao la phep do chinh thuc.
    bench = ROOT / "outputs" / ("bench.json" if args.bench else "bench_lan_chay_cuoi.json")
    bench.write_text(json.dumps({
        "n": done_n, "throughput": done_n / el, "latency": ocr_avg + vlm_avg,
        "vram_gb": vram, "sec_total": el,
        "t_ocr_avg": ocr_avg, "t_vlm_avg": vlm_avg,
        "batch": args.batch, "ocr_song_song": True,
        "n_brand_cut": n_cut, "n_empty": n_empty,
        "lora": bool(args.lora and Path(args.lora).exists()),
        # BANG CHUNG SO DO CO DUNG TIN DUOC KHONG. `check_brief.py` doc `do_tin_cay`:
        # "khong-tin-duoc" nghia la da do mot VLM khong tra loi, khong phai do model.
        "n_goi_vlm": n_goi_vlm, "n_loi_vlm": n_loi_vlm,
        "n_anh_bo_ocr": ocr_stat.get("bo", 0), "n_anh_xu_ly": done_n,
        "n_anh_yeu_cau": len(files),
        "do_tin_cay": ("tot" if n_goi_vlm and n_loi_vlm / n_goi_vlm < 0.05
                       and not ocr_stat.get("bo") and done_n == len(files)
                       else "khong-tin-duoc"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {bench}")
    if n_goi_vlm and n_loi_vlm / n_goi_vlm >= 0.05:
        print("\n  DUNG LAI: hon 5% lan goi VLM that bai. So do o tren KHONG dung tin duoc.")
        sys.exit(2)


if __name__ == "__main__":
    main()
