#!/usr/bin/env bash
# CAT 2 — one-shot demo: rebuild every binary, run the full benchmark sweep,
# validate outputs vs the seq reference, regenerate dashboard plots.
#
# Run from project root inside WSL2 Ubuntu:
#     bash bench/run_all.sh
#
# Notes:
#   * CUDA must be re-run separately in cuda/cuda_pipeline.ipynb on Colab.
#     This script preserves an existing results/cuda.csv so the dashboard still
#     includes GPU numbers.
#   * The image input.jpg in the project root is the workload (6000x3376 grayscale).

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

INPUT="${INPUT:-input.jpg}"
WARMUP="${WARMUP:-2}"
TIMED="${TIMED:-5}"
CSV="results/all.csv"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found in project root" >&2
    exit 1
fi

echo "===================================================================="
echo "CAT 2 — running full sweep"
echo "  input    : $INPUT"
echo "  warmup   : $WARMUP   timed: $TIMED"
echo "  results  : $CSV"
echo "===================================================================="

# 1. Preserve cuda.csv (rebuilt separately in Colab); start a fresh all.csv
mkdir -p results data/reference
[[ -f results/cuda.csv ]] && cp results/cuda.csv results/cuda.csv.keep
rm -f "$CSV"
# Re-stash cuda.csv (it's not part of the CPU sweep's CSV)
[[ -f results/cuda.csv.keep ]] && mv results/cuda.csv.keep results/cuda.csv

# 2. Clean rebuild
echo
echo "--- make clean && build all ---"
make clean
make seq
make omp
make pthread
make mpi

# 3. Sequential reference (also produces out_<filter>.png in data/reference/)
echo
echo "--- seq baseline ---"
./build/seq "$INPUT" all data/reference/out.png "$WARMUP" "$TIMED" "$CSV"

# 4. OpenMP sweep across {1,2,4,8} threads x {static,dynamic,guided}
echo
echo "--- omp sweep (threads x schedule) ---"
./build/omp "$INPUT" all data/reference/omp.png "$WARMUP" "$TIMED" "$CSV" sweep static

# 5. POSIX threads sweep
echo
echo "--- pthread sweep ---"
./build/pthread "$INPUT" all data/reference/pt.png "$WARMUP" "$TIMED" "$CSV" sweep

# 6. MPI sweep — ranks 1, 2, 4, 8 (oversubscribed past physical cores)
echo
echo "--- mpi sweep ---"
for N in 1 2 4 8; do
    echo "    ranks: $N"
    mpirun --oversubscribe -n "$N" ./build/mpi_pipeline \
        "$INPUT" all data/reference/mpi.png "$WARMUP" "$TIMED" "$CSV"
done

# 7. Cross-paradigm correctness (PSNR > 40 dB AND maxDiff <= 2.0)
echo
echo "--- correctness validation ---"
.venv/bin/python bench/correctness.py data/reference data/reference --min-psnr 40

# 8. Dashboard
echo
echo "--- dashboard ---"
.venv/bin/python bench/dashboard.py

echo
echo "===================================================================="
echo "Done. Plots in results/, summary in results/06_summary_table.txt"
echo "===================================================================="
