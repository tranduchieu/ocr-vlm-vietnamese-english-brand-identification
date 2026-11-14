#!/usr/bin/env python3
"""Patch manuscript Table V and derived narrative numbers from table_v.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: apply_table_v_to_tex.py table_v.json manuscript.tex")
        return 2
    table = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    tex_path = Path(sys.argv[2])
    tex = tex_path.read_text(encoding="utf-8")

    rows = []
    for code in ("B0", "B1", "L0", "M0", "M1", "M2", "M3"):
        t = table[code]
        rows.append(
            f"{code} & {fmt(t['brand_p'])} & {fmt(t['brand_r'])} & {fmt(t['brand_f1'])} & "
            f"{fmt(t['product_f1'])} & {fmt(t['ubi'])} \\\\"
        )
    block = "\n".join(rows)
    pattern = r"(\\begin\{table\*\}\[!t\]\s*\\caption\{Overall Quantitative Results\}.*?\\hline\n)(.*?)(\\end\{table\*\})"
    repl = r"\1" + "Configuration & Brand P & Brand R & Brand F1 & Product F1 & UBI \\ \\hline\n" + block + "\n" + r"\3"
    tex, n = re.subn(pattern, repl, tex, count=1, flags=re.DOTALL)
    if n != 1:
        print("WARN: main results table block not patched", n)

    b0 = table["B0"]["brand_f1"]
    m2 = table["M2"]["brand_f1"]
    l0 = table["L0"]["brand_f1"]
    m0 = table["M0"]["brand_f1"]
    m3 = table["M3"]["brand_f1"]
    m1 = table["M1"]
    abs_gain = m2 - b0
    rel = (abs_gain / b0 * 100) if b0 else 0
    l0_gain = l0 - b0
    ocr_gain = m0 - l0
    ver_gain = m3 - m0

    tex = re.sub(
        r"M2 and M3 achieve a Brand F1 of [0-9.]+, compared with [0-9.]+ for the original",
        f"M2 and M3 achieve a Brand F1 of {fmt(m2)}, compared with {fmt(b0)} for the original",
        tex,
    )
    tex = re.sub(
        r"absolute increase of [0-9.]+, or approximately [0-9.]+% relative",
        f"absolute increase of {fmt(abs_gain)}, or approximately {rel:.2f}% relative",
        tex,
    )
    tex = re.sub(
        r"LoRA adaptation increases Brand F1 by [0-9.]+",
        f"LoRA adaptation increases Brand F1 by {fmt(l0_gain)}",
        tex,
    )
    tex = re.sub(
        r"LoRA adaptation improves Brand F1 from [0-9.]+ to [0-9.]+ without external OCR",
        f"LoRA adaptation improves Brand F1 from {fmt(b0)} to {fmt(l0)} without external OCR",
        tex,
    )
    tex = re.sub(
        r"Adding OCR further increases F1 to [0-9.]+ \(L0 to M0\)",
        f"Adding OCR further increases F1 to {fmt(m0)} (L0 to M0)",
        tex,
    )
    tex = re.sub(
        r"Verification yields an observed increase to [0-9.]+ \(M0 to M3\)",
        f"Verification yields an observed increase to {fmt(m3)} (M0 to M3)",
        tex,
    )
    tex = re.sub(
        r"external-OCR-only verification reduces UBI from [0-9.]+ to [0-9.]+",
        f"external-OCR-only verification reduces UBI from {fmt(table['M0']['ubi'])} to {fmt(m1['ubi'])}",
        tex,
    )
    tex = re.sub(
        r"lowering Brand Recall from [0-9.]+ to [0-9.]+",
        f"lowering Brand Recall from {fmt(table['M0']['brand_r'])} to {fmt(m1['brand_r'])}",
        tex,
    )
    tex = re.sub(
        r"restores most of this loss \(Brand Recall [0-9.]+\)",
        f"restores most of this loss (Brand Recall {fmt(table['M2']['brand_r'])})",
        tex,
    )
    tex = re.sub(
        r"although UBI rises to [0-9.]+\.",
        f"although UBI rises to {fmt(table['M2']['ubi'])}.",
        tex,
        count=1,
    )
    tex = re.sub(
        r"increases Brand F1 by only [0-9.]+, a difference",
        f"increases Brand F1 by only {fmt(table['M2']['brand_f1'] - table['M0']['brand_f1'])}, a difference",
        tex,
    )

    # verification summary table if present
    for code, key in (("M0", "m0"), ("M1", "m1"), ("M2", "m2"), ("M3", "m3")):
        t = table[code]
        tex = re.sub(
            rf"({code}:.*?&) [0-9.]+ (& [0-9.]+ \\\\)",
            rf"\1 {fmt(t['brand_f1'])} & {fmt(t['ubi'])} \\\\",
            tex,
        )

    tex_path.write_text(tex, encoding="utf-8")
    print("patched", tex_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
