# CAT 2 — Parallel Image Processing Pipeline

A demonstration project for the Parallel Systems unit. The same five image filters are implemented across **five parallel paradigms**, run on the same 20-megapixel input image, and compared on a unified dashboard.

Paradigms covered: **sequential** (C reference), **OpenMP**, **POSIX threads with manual pool**, **MPI distributed**, **CUDA on NVIDIA T4** (Colab).

---

## Why this exists

CAT 1 (`CAT 1 sequential_processing.py`) was flagged as "too simple" — pure-Python sequential, two filters, ≤480p images, no parallelism. CAT 2 addresses that by:

1. **Backfilling** the parallelism the abstract promised (OpenMP + POSIX threads — these were originally CAT 1 scope but were never delivered).
2. **Delivering** the originally-scoped CAT 2 work (MPI distributed + CUDA GPU).
3. Choosing **non-trivial parallel patterns** for each paradigm — manual thread pool, halo-region exchange, shared-memory tiling, parallel reduction+scan — not just "filter X with paradigm Y."

The lecturer's "too simple" feedback was about *depth per paradigm*, not paradigm count. Each piece of this project deliberately picks a hard bit of the parallelism story to demonstrate.

---

## What was built

### The five filters (same set in every paradigm)

| Filter | Pattern | Why included |
|---|---|---|
| `blur` (Gaussian 5×5) | Stencil | Classical halo-needing filter, halo width 2 |
| `sobel` | Stencil + global max-reduction | Adds a reduction on top of stencil |
| `sharp` (unsharp mask) | Composition (blur + subtract) | Tests pipeline composition |
| `bc` (brightness/contrast) | Pointwise | Embarrassingly parallel — overhead floor |
| `histeq` (histogram equalisation) | **Reduction + scan + map** | The deliberate non-embarrassingly-parallel filter |

`histeq` is the most important of these five. It forces a real parallel-pattern engineering exercise in every paradigm: build per-thread/rank histograms (reduction), compute the CDF (scan), then apply the LUT (map). The other filters are well-trodden territory; `histeq` shows you understand parallel patterns beyond stencils.

### What each paradigm contributes (the depth move per paradigm)

| Paradigm | File | Headline depth move |
|---|---|---|
| seq | `kernels/filters_seq.c` | Reference oracle. Every other paradigm's correctness is judged against this. |
| OpenMP | `kernels/filters_omp.c` | `schedule(runtime)` + sweep across **static / dynamic / guided** to surface scheduling impact. |
| POSIX threads | `kernels/filters_pthread.c` + `thread_pool.c` | **Persistent worker pool** with mutex+condvar work queue + `pool_wait` barrier. Filters submit row-strip tasks; histeq uses thread-local 256-bin histograms + main-thread merge. |
| MPI | `kernels/filters_mpi.c` | **Halo exchange** with `MPI_Sendrecv` between neighbours, mirrored boundaries at global edges. `MPI_Allreduce(MAX)` for sobel, `MPI_Allreduce(SUM)` for histogram. |
| CUDA | `cuda/cuda_pipeline.ipynb` | **Naive vs. tiled shared-memory** blur kernel comparison + `cudaEvent` H2D/Kernel/D2H breakdown. |

---

## Headline findings — what to actually talk about in the demo

These are the points that make this project look like real measurement work, not just "I wrote four versions of the same thing."

### 1. OpenMP `schedule(dynamic)` with default chunk=1 is catastrophic on fine loops
For `bc` (pointwise) and `histeq` it's **7×–24× slower** than `static` — and it actively gets *worse* with more threads. This is visible on `results/04_omp_schedule_comparison.png` as the orange line sitting an order of magnitude above the others and trending UP. The OpenMP runtime is shipping every individual loop iteration through a synchronization channel. This is exactly the kind of "the textbook didn't warn me about this" finding that wins marks.

### 2. CUDA naive blur is *faster* than tiled-shared-memory blur on T4
The textbook claim — "tile into shared memory to amortise global reads" — does not hold for a 5×5 stencil on a Turing-class GPU. Numbers from `results/cuda.csv`:
- naive kernel: **1.81 ms**
- tiled kernel: **2.34 ms** (0.77× — slower)

Reason: T4's L2 cache absorbs the redundant global reads from the naive kernel, and the cooperative tile-load + `__syncthreads()` overhead exceeds the savings on a stencil this small. The tiled approach would dominate on larger stencils (9×9, 11×11) or older GPUs without aggressive caching. **Measurement contradicts intuition.**

### 3. MPI sweet spot is at p = physical core count (4), not logical (8)
Every filter regresses or plateaus at p=8 on this 4-physical-core machine. `histeq` actually gets *worse* with more ranks (1.10× speedup at p=4, slowdown at p=8). The reason: scatter+gather of a 20-megapixel image plus `MPI_Allreduce` overhead dwarfs the ~40ms of actual histogram work. **Don't parallelise small workloads with high-overhead paradigms** — and now you have hard numbers to prove it, not just an abstract claim.

### 4. OpenMP and pthreads come out roughly equal, but for opposite reasons
On `histeq`, OpenMP wins (libgomp's `reduction(+:array)` is hand-tuned). On `sharp`, pthreads wins (manual strip balance avoids OMP's runtime overhead). Plot 5 (`05_omp_vs_pthread.png`) shows this directly. The story: **OpenMP for productivity, pthreads when you need to control the partitioning yourself.** Same speedup ceiling, different ergonomics.

### 5. CUDA pays 75–80% of its time in transfers, not compute
The CUDA breakdown chart (`results/03_cuda_breakdown.png`) shows H2D + D2H dominating every filter. Kernel time is 1–4 ms; PCIe transfer time is 8 ms regardless of filter complexity. **GPU economics aren't about kernel speed — they're about whether you can amortise the transfer.** This motivates batching, kernel fusion, and keeping data on the device between filters (none of which we did, but the chart explains *why* you would).

### 6. All 106 parallel outputs are bit-identical to the seq reference
PSNR=∞, maxDiff=0 on every OpenMP / pthread / MPI / CUDA output vs. `data/reference/out_<filter>.png`. The dual-threshold validation (PSNR > 40 dB AND maxDiff ≤ 2.0) caught a real bug during MPI development — an off-by-one in the global-boundary mirror that gave PSNR=69 dB (looked fine) but maxDiff=128 (very broken). PSNR-only validation would have shipped a broken implementation.

---

## How to demo this

### One-command rebuild + sweep + dashboard (CPU paradigms)

```bash
bash bench/run_all.sh
```

This runs from a clean state: `make clean`, builds all four CPU binaries, sweeps every paradigm × thread/rank count × filter, validates correctness, and regenerates every plot. Takes ~5–10 minutes on this machine.

### CUDA (separate, runs in Colab)

CUDA cannot run locally (no NVIDIA GPU on this laptop). Instead:

1. Open `cuda/cuda_pipeline.ipynb` at https://colab.research.google.com (`File → Upload notebook`).
2. `Runtime → Change runtime type → GPU` (T4 is fine).
3. Run cells in order. Cell 4 will prompt you to upload `input.jpg` and the 5 `data/reference/out_*.png` reference files.
4. The final cell zips outputs (`cuda_outputs.zip`) and downloads them. Drop into `results/cuda_outputs/` and copy `cuda.csv` and `cuda_*.png` into `results/` and `data/reference/` respectively.

The CUDA notebook is generated by `cuda/build_notebook.py` — edit *that* if you need to change kernels, then re-run `python cuda/build_notebook.py`. Don't hand-edit the .ipynb.

### What to show on screen during the demo

In order:
1. `cat "CAT 1 sequential_processing.py" | wc -l` — ~110 lines, one paradigm. Then `find kernels -name '*.c' | xargs wc -l` — substantially more across five paradigms.
2. `bash bench/run_all.sh` — let the sweep run. Narrate the paradigms as their output streams.
3. `cat results/06_summary_table.txt` — best-paradigm-per-filter, best-speedup-per-paradigm, the naive-vs-tiled CUDA finding.
4. Open `results/04_omp_schedule_comparison.png` — point at the orange line. Tell the dynamic-chunk-1 story.
5. Open `results/03_cuda_breakdown.png` — point at the H2D+D2H bars. Tell the GPU-economics story.
6. Open `results/02_speedup_per_filter.png` — show the speedup curves vs ideal-linear. Note where each paradigm wins.
7. Run correctness one more time: `.venv/bin/python bench/correctness.py data/reference data/reference --min-psnr 40` — show 100+/100+ pass.

---

## Project layout

```
.
├── kernels/                        # Shared C kernel library
│   ├── filters.h                   # image_t struct, MIRROR macro, all signatures
│   ├── filters_seq.c               # Reference (correctness oracle)
│   ├── filters_omp.c               # OpenMP versions
│   ├── filters_pthread.c           # POSIX threads versions
│   ├── filters_mpi.c               # MPI versions with halo helpers
│   ├── thread_pool.{h,c}           # Reusable mutex+condvar worker pool
│   ├── image_io.{h,c}              # JPG/PNG load/save (stb_image, vendored)
│   ├── bench.{h,c}                 # clock_gettime timer + warmup/timed protocol
│   ├── main_seq.c, main_omp.c, main_pthread.c, main_mpi.c   # Per-paradigm drivers
│   └── stb_image.h, stb_image_write.h    # Public-domain image I/O
├── cuda/
│   ├── cuda_pipeline.ipynb         # Colab notebook (run on T4)
│   └── build_notebook.py           # Generator for the notebook
├── bench/
│   ├── correctness.py              # PSNR + maxDiff cross-paradigm validator
│   ├── dashboard.py                # Generates the 5 plots + summary table
│   └── run_all.sh                  # One-shot demo
├── data/
│   └── reference/                  # Sequential reference outputs + all paradigm outputs
├── results/                        # CSVs + PNG charts + summary text
│   ├── all.csv                     # 105 rows: seq + omp + pthread + mpi
│   ├── cuda.csv                    # 6 rows from the CUDA notebook
│   ├── 01_walltime_per_filter.png
│   ├── 02_speedup_per_filter.png
│   ├── 03_cuda_breakdown.png
│   ├── 04_omp_schedule_comparison.png
│   ├── 05_omp_vs_pthread.png
│   └── 06_summary_table.txt
├── input.jpg                       # The 6000×3376 (~20MP) workload image
├── Makefile                        # `make seq omp pthread mpi`
├── CLAUDE.md                       # Notes for future Claude Code sessions
└── README.md                       # This file
```

---

## Honest scope notes — what was *not* done, and why

- **Single image instead of an 8–16 image batch at multiple resolutions.** The original plan called for batches at 1080p+4K. We used one 20-megapixel image. This is large enough that the headline findings (transfer overhead, scheduling effects, halo cost) are all visible. Adding batch processing would require modifying every driver and re-running every sweep, with no change in the qualitative story. Mention as future work in the report.
- **OpenCilk paradigm dropped.** The abstract mentioned OpenCilk; we skipped it. It's a Linux-only research compiler with painful installation, and OpenMP + pthreads already cover the shared-memory C story. Java threads + Python threads were also dropped for the same reason — they'd add filenames without adding insight.
- **Single-node MPI only.** Multi-node would prove the pipeline crosses processes properly but adds VM setup time. The single-node sweep already shows the comm-vs-compute economics.
- **CUDA "third kernel version" (coalesced + bank-conflict-free) dropped.** Per "keep it simple" — naive vs tiled is enough to make the depth-move point. The surprise finding (naive wins on T4) is more pedagogically valuable than three nearly-identical tiled variants.

---

## What's the demo's "wow" actually?

If you have to describe the project in one sentence: **"Five parallelism paradigms, same five filters, same image, fully validated bit-identical, with measurements that contradict three textbook claims."** The findings list above is the wow. The plots are the evidence. The 100+/100+ PSNR-passing outputs are the credibility check.

If you're asked "how is this different from CAT 1?" — show the line counts, show the schedule-comparison plot, show the CUDA breakdown chart, show the summary table. CAT 1 was a single sequential script that took a couple of hours. CAT 2 is a measured comparison across five hardware-and-runtime models with empirical findings the textbook doesn't warn you about.
