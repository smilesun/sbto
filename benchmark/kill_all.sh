#!/usr/bin/env bash
# Kill all running benchmark and sbto/main.py processes.
PIDS=$(ps aux | grep -E "sbto/main\.py|launch_benchmark|benchmark_pcbo|benchmark_sv|benchmark_wishart" \
    | grep -v grep | awk '{print $2}')
if [ -z "$PIDS" ]; then
    echo "No benchmark processes running."
else
    echo "Killing PIDs: $PIDS"
    kill $PIDS
    echo "Done."
fi
