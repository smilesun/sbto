#!/usr/bin/env bash
# Show progress of all benchmark JSON results and any running benchmark processes.
# Usage (from anywhere):  bash benchmark/status.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCHMARK_DIR="$REPO_ROOT/benchmark_rst_json_pointers"

print_json() {
    local label="$1"
    local file="$2"
    echo "=== $label ==="
    if [ -f "$file" ]; then
        python3 -c "
import json, sys
d = json.load(open('$file'))
for r in d:
    rc   = r.get('returncode', '?')
    nc   = str(r.get('n_clusters', '?')).rjust(3)
    cost = r.get('best_cost', None)
    cost_str = ('%.1f' % cost) if cost is not None else 'None'
    ok = ' <1k' if cost is not None and cost < 1000 else ''
    solver = r['params'].get('solver', 'pcbo')
    label = r['params'].get('label', r['trial_id'])
    print(f\"  {r['trial_id']}  {solver:<10}  {label:<30}  rc={rc}  n_clust={nc}  best_cost={cost_str}{ok}\")
print(f'  Completed: {len(d)}')
"
    else
        echo "  (no results yet)"
    fi
    echo ""
}

# Auto-scan all JSON files in benchmark_rst_json_pointers/
shopt -s nullglob
json_files=("$BENCHMARK_DIR"/*.json)
shopt -u nullglob

if [ ${#json_files[@]} -eq 0 ]; then
    echo "(no result files in $BENCHMARK_DIR)"
else
    for f in "${json_files[@]}"; do
        print_json "$(basename "$f" .json)" "$f"
    done
fi

echo "=== Running benchmark processes ==="
PROCS=$(ps aux | grep -E "launch_benchmark|sbto/main\.py" | grep -v grep)
if [ -n "$PROCS" ]; then
    echo "$PROCS" | awk '{printf "  PID=%-8s  %s %s %s\n", $2, $11, $12, $13}' | head -10
else
    echo "  (none running)"
fi
echo ""

echo "=== Log files ==="
LOG_DIR="$REPO_ROOT/benchmark_logs"
shopt -s nullglob
logs=("$LOG_DIR"/*.log)
shopt -u nullglob
if [ ${#logs[@]} -gt 0 ]; then
    for f in "${logs[@]}"; do
        lines=$(wc -l < "$f" 2>/dev/null || echo 0)
        last=$(tail -1 "$f" 2>/dev/null | cut -c1-80)
        echo "  $(basename "$f")  ($lines lines)  last: $last"
    done
else
    echo "  (no logs in benchmark_logs/)"
fi
