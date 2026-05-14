#!/usr/bin/env bash
# Run all three benchmarks back-to-back and write results to bench_results.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -d .venv ]] && source .venv/bin/activate

export PYTHONPATH="$REPO_ROOT/qwen_megakernel:$REPO_ROOT:${PYTHONPATH:-}"
export LDG_VOCAB_SIZE="${LDG_VOCAB_SIZE:-2052}"

OUT="${BENCH_OUT:-bench_results.md}"

{
    echo "# Benchmark results"
    echo
    echo "Generated: $(date -u +%FT%TZ)"
    echo
    echo "## GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    echo
    echo "## 1. Megakernel throughput (talker weights)"
    echo '```'
    LDG_VOCAB_SIZE=3072 python -m benchmarks.throughput --model talker --tokens 100 --runs 20
    echo '```'
    echo
    echo "## 2. Megakernel throughput (base Qwen3-0.6B, sanity)"
    echo '```'
    LDG_VOCAB_SIZE=151936 python -m benchmarks.throughput --model base --tokens 100 --runs 20
    echo '```'
    echo
    echo "## 3. TTFC (local — engine only)"
    echo '```'
    python -m benchmarks.ttfc --mode local --runs 10
    echo '```'
    echo
    echo "## 4. RTF (end-to-end)"
    echo '```'
    python -m benchmarks.rtf --runs 5
    echo '```'
} | tee "$OUT"

echo
echo "Results written to $OUT"
