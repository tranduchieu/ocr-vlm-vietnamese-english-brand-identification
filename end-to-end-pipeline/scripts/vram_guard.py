#!/usr/bin/env python3
"""Chay mot lenh Python duoi TRAN VRAM CUNG (mac dinh 12.5 GiB cho tien trinh nay).

Hai tang bao ve:
  1. Trong tien trinh con: torch.cuda.set_per_process_memory_fraction -> allocator nem
     OOM TRUOC khi vuot tran (khong de GPU bi nhoi qua muc).
  2. Watchdog ngoai: doc `nvidia-smi --query-compute-apps` theo PID (gom ca cay tien trinh
     con, gom ca CUDA context + paddle/vietocr), neu tong > KILL_GIB thi SIGKILL ngay.

Chi dem VRAM cua CHINH tien trinh nay, khong dem Desktop/tien trinh nguoi khac.

Dung:
  python scripts/vram_guard.py [--cap-gib 12.5] [--log FILE] -- <script.py> [args...]
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

CHILD_BOOT = r"""
import os, runpy, sys
import torch
cap = float(os.environ["VRAM_GUARD_TORCH_CAP_GIB"])
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
torch.cuda.set_per_process_memory_fraction(min(0.99, cap / total), 0)
print(f"[vram_guard] torch allocator cap = {cap:.2f} GiB / {total:.2f} GiB", flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


def descendants(pid: int) -> set[int]:
    out = {pid}
    try:
        txt = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True).stdout
    except Exception:
        return out
    kids: dict[int, list[int]] = {}
    for line in txt.splitlines():
        try:
            p, pp = map(int, line.split())
        except ValueError:
            continue
        kids.setdefault(pp, []).append(p)
    stack = [pid]
    while stack:
        cur = stack.pop()
        for k in kids.get(cur, []):
            if k not in out:
                out.add(k)
                stack.append(k)
    return out


def used_mib(pids: set[int]) -> int:
    try:
        txt = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return 0
    tot = 0
    for line in txt.splitlines():
        try:
            p, m = [x.strip() for x in line.split(",")]
            if int(p) in pids:
                tot += int(m)
        except ValueError:
            continue
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gib", type=float, default=12.5, help="tran cung cua tien trinh")
    ap.add_argument("--torch-margin-gib", type=float, default=1.5,
                    help="allocator torch bi chan o cap - margin (chua cho CUDA context)")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("thieu lenh sau --")

    env = dict(os.environ)
    env["VRAM_GUARD_TORCH_CAP_GIB"] = f"{a.cap_gib - a.torch_margin_gib:.3f}"
    kill_mib = int(a.cap_gib * 1024) - 100     # kill truoc tran 100 MiB cho chac
    child = subprocess.Popen([sys.executable, "-c", CHILD_BOOT, *cmd], env=env,
                             start_new_session=True)
    state = {"peak": 0, "killed": False, "stop": False}

    def watch() -> None:
        lf = open(a.log, "w") if a.log else None
        t0 = time.time()
        while not state["stop"]:
            m = used_mib(descendants(child.pid))
            if m > state["peak"]:
                state["peak"] = m
            if lf:
                lf.write(f"{time.time()-t0:.1f}\t{m}\t{state['peak']}\n")
                lf.flush()
            if m > kill_mib:
                state["killed"] = True
                print(f"\n[vram_guard] !!! {m} MiB > {kill_mib} MiB -> KILL", flush=True)
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(a.poll)
        if lf:
            lf.close()

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        rc = child.wait()
    except KeyboardInterrupt:
        os.killpg(child.pid, signal.SIGTERM)
        rc = child.wait()
    state["stop"] = True
    th.join(timeout=3)
    print(f"\n[vram_guard] DINH VRAM tien trinh = {state['peak']} MiB "
          f"({state['peak']/1024:.2f} GiB) · tran {a.cap_gib} GiB · "
          f"{'BI KILL' if state['killed'] else 'khong vuot tran'} · exit={rc}", flush=True)
    return 99 if state["killed"] else rc


if __name__ == "__main__":
    sys.exit(main())
