#!/usr/bin/env python3
"""CAT 2 dashboard: produce comparative plots from results/all.csv + results/cuda.csv.

Run from project root:
    .venv/bin/python bench/dashboard.py

Outputs PNGs into results/:
    01_walltime_per_filter.png
    02_speedup_per_filter.png
    03_cuda_breakdown.png
    04_omp_schedule_comparison.png
    05_omp_vs_pthread.png
    06_summary_table.txt
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILTERS = ["blur", "sobel", "sharp", "bc", "histeq"]
RESULTS = Path("results")

PARADIGM_STYLE = {
    "seq":         {"color": "#888888", "marker": "x", "label": "seq (1 thread)"},
    "omp_static":  {"color": "#1f77b4", "marker": "o", "label": "omp static"},
    "omp_dynamic": {"color": "#ff7f0e", "marker": "s", "label": "omp dynamic"},
    "omp_guided":  {"color": "#2ca02c", "marker": "^", "label": "omp guided"},
    "pthread":     {"color": "#9467bd", "marker": "D", "label": "pthread"},
    "mpi":         {"color": "#d62728", "marker": "v", "label": "mpi"},
}


def load_csvs():
    all_csv = RESULTS / "all.csv"
    cuda_csv = RESULTS / "cuda.csv"
    if not all_csv.exists():
        print(f"ERROR: {all_csv} not found", file=sys.stderr)
        sys.exit(1)
    df_cpu = pd.read_csv(all_csv)
    df_cuda = pd.read_csv(cuda_csv) if cuda_csv.exists() else None
    return df_cpu, df_cuda


def plot_walltime_per_filter(df_cpu, df_cuda, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=False)
    for ax, filt in zip(axes, FILTERS):
        sub = df_cpu[df_cpu["filter"] == filt]
        for para, style in PARADIGM_STYLE.items():
            d = sub[sub.paradigm == para].sort_values("threads")
            if len(d) == 0:
                continue
            ax.plot(d.threads, d.mean_s * 1000, **{k: v for k, v in style.items() if k != "label"},
                    label=style["label"], linewidth=1.4, markersize=5)
        if df_cuda is not None:
            cuda_filt = "blur_tiled" if filt == "blur" else filt
            row = df_cuda[df_cuda["filter"] == cuda_filt]
            if len(row):
                ax.axhline(row.mean_s.iloc[0] * 1000, color="black", linestyle="--",
                           linewidth=1.0, label="cuda T4")
        ax.set_title(filt)
        ax.set_xlabel("threads / ranks")
        ax.set_ylabel("mean wall-clock (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Wall-clock per filter — all paradigms (lower is better, log-log)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_speedup_per_filter(df_cpu, out_path):
    """Speedup = T(N=1, same paradigm) / T(N). Lets each paradigm be its own baseline,
    avoids cross-paradigm thermal/load contamination."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=False)
    for ax, filt in zip(axes, FILTERS):
        sub = df_cpu[df_cpu["filter"] == filt]
        max_n = 1
        for para, style in PARADIGM_STYLE.items():
            if para == "seq":
                continue
            d = sub[sub.paradigm == para].sort_values("threads")
            if len(d) == 0:
                continue
            t1 = d[d.threads == 1]
            if len(t1) == 0:
                continue
            base = t1.mean_s.iloc[0]
            d = d.assign(speedup=base / d.mean_s)
            ax.plot(d.threads, d.speedup, **{k: v for k, v in style.items() if k != "label"},
                    label=style["label"], linewidth=1.4, markersize=5)
            max_n = max(max_n, d.threads.max())
        # Ideal linear speedup line
        xs = [1, max_n] if max_n > 1 else [1, 8]
        ax.plot(xs, xs, color="black", linestyle=":", linewidth=1.0, label="ideal (linear)")
        ax.set_title(filt)
        ax.set_xlabel("threads / ranks (N)")
        ax.set_ylabel("speedup vs N=1 of same paradigm")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.grid(True, alpha=0.3, which="both")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Speedup curves per filter (paradigm-internal baseline). Above 1 = real gain", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cuda_breakdown(df_cuda, out_path):
    if df_cuda is None or df_cuda.empty:
        print("  skip cuda breakdown — no cuda.csv")
        return
    plot_df = df_cuda[df_cuda["filter"] != "blur_naive"].set_index("filter")
    order = ["blur_tiled", "sobel", "sharp", "bc", "histeq"]
    order = [f for f in order if f in plot_df.index]
    plot_df = plot_df.reindex(order)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df.h2d_ms,    label="H2D",    color="#d97757")
    ax.bar(x, plot_df.kernel_ms, bottom=plot_df.h2d_ms, label="Kernel", color="#3a8d99")
    ax.bar(x, plot_df.d2h_ms,    bottom=plot_df.h2d_ms + plot_df.kernel_ms, label="D2H", color="#a4c763")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=10)
    ax.set_ylabel("Time (ms, mean of 10)")
    ax.set_title("CUDA per-filter breakdown — H2D + Kernel + D2H (T4 GPU)")
    ax.legend()
    for i, (f, row) in enumerate(plot_df.iterrows()):
        total = row.h2d_ms + row.kernel_ms + row.d2h_ms
        transfer_pct = 100 * (row.h2d_ms + row.d2h_ms) / total
        ax.text(i, total + 0.2, f"{transfer_pct:.0f}% transfer",
                ha="center", va="bottom", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_omp_schedule(df_cpu, out_path):
    omp_paradigms = ["omp_static", "omp_dynamic", "omp_guided"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=False)
    for ax, filt in zip(axes, FILTERS):
        sub = df_cpu[df_cpu["filter"] == filt]
        for para in omp_paradigms:
            d = sub[sub.paradigm == para].sort_values("threads")
            if len(d) == 0:
                continue
            style = PARADIGM_STYLE[para]
            ax.plot(d.threads, d.mean_s * 1000, **{k: v for k, v in style.items() if k != "label"},
                    label=style["label"], linewidth=1.4, markersize=5)
        ax.set_title(filt)
        ax.set_xlabel("threads")
        ax.set_ylabel("mean wall-clock (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("OpenMP scheduling comparison: static vs. dynamic vs. guided", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_omp_vs_pthread(df_cpu, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=False)
    for ax, filt in zip(axes, FILTERS):
        sub = df_cpu[df_cpu["filter"] == filt]
        for para in ["omp_static", "pthread"]:
            d = sub[sub.paradigm == para].sort_values("threads")
            if len(d) == 0:
                continue
            style = PARADIGM_STYLE[para]
            ax.plot(d.threads, d.mean_s * 1000, **{k: v for k, v in style.items() if k != "label"},
                    label=style["label"], linewidth=1.6, markersize=6)
        ax.set_title(filt)
        ax.set_xlabel("threads")
        ax.set_ylabel("mean wall-clock (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("OpenMP (static) vs. POSIX threads — productivity vs. low-level control", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary(df_cpu, df_cuda, out_path):
    """One-page text summary: best-time-per-filter and best-speedup-per-paradigm."""
    lines = []
    lines.append("CAT 2 — summary table\n")
    lines.append("=" * 78 + "\n")
    lines.append(f"Source rows: {len(df_cpu)} CPU + "
                 f"{len(df_cuda) if df_cuda is not None else 0} CUDA\n\n")

    lines.append("Best wall-clock per filter (across all paradigms+configs)\n")
    lines.append("-" * 78 + "\n")
    lines.append(f"{'filter':<8}  {'best paradigm':<18}  {'N':>3}  {'mean (ms)':>10}\n")
    for filt in FILTERS:
        sub = df_cpu[df_cpu["filter"] == filt]
        if df_cuda is not None:
            cuda_filt = "blur_tiled" if filt == "blur" else filt
            crow = df_cuda[df_cuda["filter"] == cuda_filt]
            if len(crow):
                sub = pd.concat([sub, crow.assign(filter=filt)], ignore_index=True)
        if len(sub) == 0:
            continue
        best = sub.loc[sub.mean_s.idxmin()]
        lines.append(f"{filt:<8}  {best.paradigm:<18}  {int(best.threads):>3}  "
                     f"{best.mean_s * 1000:>10.3f}\n")

    lines.append("\nBest speedup per paradigm vs its own t=1 baseline\n")
    lines.append("-" * 78 + "\n")
    lines.append(f"{'paradigm':<14}  {'filter':<8}  {'speedup':>8}  {'N':>3}\n")
    for para in ["omp_static", "omp_dynamic", "omp_guided", "pthread", "mpi"]:
        sub = df_cpu[df_cpu.paradigm == para]
        if len(sub) == 0:
            continue
        best_record = None
        for filt in FILTERS:
            f_sub = sub[sub["filter"] == filt]
            t1 = f_sub[f_sub.threads == 1]
            if len(t1) == 0 or len(f_sub) == 0:
                continue
            base = t1.mean_s.iloc[0]
            f_sub = f_sub.assign(speedup=base / f_sub.mean_s)
            row = f_sub.loc[f_sub.speedup.idxmax()]
            if best_record is None or row.speedup > best_record.speedup:
                best_record = row
        if best_record is not None:
            lines.append(f"{para:<14}  {best_record['filter']:<8}  "
                         f"{best_record.speedup:>7.2f}x  {int(best_record.threads):>3}\n")

    if df_cuda is not None and "blur_naive" in df_cuda["filter"].values and "blur_tiled" in df_cuda["filter"].values:
        lines.append("\nCUDA depth-move: naive vs tiled blur kernel\n")
        lines.append("-" * 78 + "\n")
        n = df_cuda[df_cuda["filter"] == "blur_naive"].iloc[0]
        t = df_cuda[df_cuda["filter"] == "blur_tiled"].iloc[0]
        lines.append(f"  naive kernel : {n.kernel_ms:.3f} ms\n")
        lines.append(f"  tiled kernel : {t.kernel_ms:.3f} ms\n")
        ratio = n.kernel_ms / t.kernel_ms
        verdict = "tiled wins" if ratio > 1 else "naive wins (T4 L2 cache absorbs redundant reads)"
        lines.append(f"  ratio        : {ratio:.2f}x  ({verdict})\n")

    out_path.write_text("".join(lines))
    print(f"  wrote {out_path}")
    print("\n" + "".join(lines))


def main():
    df_cpu, df_cuda = load_csvs()
    print(f"Loaded {len(df_cpu)} CPU rows + {0 if df_cuda is None else len(df_cuda)} CUDA rows")
    print(f"Paradigms: {sorted(df_cpu.paradigm.unique())}")

    plot_walltime_per_filter(df_cpu, df_cuda, RESULTS / "01_walltime_per_filter.png")
    plot_speedup_per_filter(df_cpu, RESULTS / "02_speedup_per_filter.png")
    plot_cuda_breakdown(df_cuda, RESULTS / "03_cuda_breakdown.png")
    plot_omp_schedule(df_cpu, RESULTS / "04_omp_schedule_comparison.png")
    plot_omp_vs_pthread(df_cpu, RESULTS / "05_omp_vs_pthread.png")
    write_summary(df_cpu, df_cuda, RESULTS / "06_summary_table.txt")
    print("Done.")


if __name__ == "__main__":
    main()
