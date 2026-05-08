#!/usr/bin/env bash
# Generic background benchmark launcher.
# Usage (from repo root):
#   bash benchmark/launch.sh benchmark_recipes/overnight_sv_vs_pcbo.yaml
#   bash benchmark/launch.sh benchmark_recipes/pcbo_search.yaml --resume

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -eq 0 ]; then
    echo "Usage: bash benchmark/launch.sh <config.yaml> [--resume]"
    exit 1
fi

CONFIG="$1"
shift
ARGS="${*}"

STEM="$(basename "$CONFIG" .yaml)"
LOG="benchmark_logs/${STEM}.log"
mkdir -p benchmark_logs

echo "Config : $CONFIG"
echo "Results: benchmark_rst_json_pointers/${STEM}.json"
echo "Log    : $LOG"
echo ""

nohup uv run python benchmark/launch_benchmark.py "$CONFIG" $ARGS \
    > "$LOG" 2>&1 &
PID=$!

echo "PID=$PID — benchmark running in background."
echo "Monitor: tail -f $LOG"
echo "Status : bash benchmark/status.sh"
