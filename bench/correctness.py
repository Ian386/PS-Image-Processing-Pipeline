#!/usr/bin/env python3
"""PSNR validator: compares paradigm outputs against the sequential reference.

Usage:
    python3 bench/correctness.py <reference_dir> <candidate_dir> [--filters blur,sobel,...] [--min-psnr 40]

Convention: reference files are named out_<filter>.png in <reference_dir>.
Candidate files match glob *_<filter>_*.png OR <prefix>_<filter>.png in <candidate_dir>.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

DEFAULT_FILTERS = ["blur", "sobel", "sharp", "bc", "histeq"]


def load_grey(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def psnr(a, b):
    if a.shape != b.shape:
        return None, None
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf"), 0.0
    max_diff = float(np.max(np.abs(a - b)))
    return 20.0 * np.log10(255.0 / np.sqrt(mse)), max_diff


def find_candidates(cand_dir, filter_name):
    p = Path(cand_dir)
    matches = sorted(set(list(p.glob(f"*_{filter_name}_*.png")) + list(p.glob(f"*_{filter_name}.png"))))
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_dir")
    ap.add_argument("candidate_dir")
    ap.add_argument("--filters", default=",".join(DEFAULT_FILTERS))
    ap.add_argument("--min-psnr", type=float, default=40.0)
    args = ap.parse_args()

    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    ref_dir = Path(args.reference_dir)
    cand_dir = Path(args.candidate_dir)

    failures = 0
    total = 0
    print(f"{'filter':<10} {'candidate':<55} {'PSNR(dB)':>10} {'maxDiff':>8}  status")
    print("-" * 100)

    for filt in filters:
        ref_path = ref_dir / f"out_{filt}.png"
        if not ref_path.exists():
            print(f"{filt:<10} reference missing: {ref_path}")
            failures += 1
            continue
        ref = load_grey(ref_path)

        candidates = find_candidates(cand_dir, filt)
        if not candidates:
            print(f"{filt:<10} no candidates found in {cand_dir} for filter '{filt}'")
            continue

        for c in candidates:
            if c.name == ref_path.name and ref_path.parent == c.parent:
                continue
            cand = load_grey(c)
            p, mx = psnr(ref, cand)
            total += 1
            if p is None:
                print(f"{filt:<10} {c.name:<55}      shape mismatch")
                failures += 1
                continue
            ok = (p >= args.min_psnr) and (mx <= 2.0)
            tag = "OK" if ok else "FAIL"
            if not ok:
                failures += 1
            ps = "inf" if p == float("inf") else f"{p:.2f}"
            print(f"{filt:<10} {c.name:<55} {ps:>10} {mx:>8.2f}  {tag}")

    print("-" * 100)
    print(f"{total - failures}/{total} pass; threshold PSNR>={args.min_psnr}dB AND maxDiff<=2.0")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
