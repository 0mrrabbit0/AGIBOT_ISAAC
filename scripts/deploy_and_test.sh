#!/bin/bash
# Deploy scripted sorting policy to Docker and run tests.
#
# Usage: bash scripts/deploy_and_test.sh [diagnose|benchmark]
#   diagnose  - Run IK/FK diagnostic only (no Isaac Sim needed)
#   benchmark - Run the full sorting benchmark in Isaac Sim

set -e

CONTAINER="uwlab_train"
WORKSPACE="/workspace/genie_sim_RL"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

echo "=== Deploying ScriptedSortingPolicy to Docker ==="

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: Docker container '${CONTAINER}' is not running."
    echo "Start it first, then re-run this script."
    exit 1
fi

# Create workspace directory in container
docker exec "$CONTAINER" mkdir -p "$WORKSPACE/scripts"

# Copy policy files
echo "Copying policy files..."
docker cp "$SCRIPTS_DIR/scripted_sorting_policy.py" \
    "$CONTAINER:$WORKSPACE/scripts/scripted_sorting_policy.py"
docker cp "$SCRIPTS_DIR/run_sorting_benchmark.py" \
    "$CONTAINER:$WORKSPACE/scripts/run_sorting_benchmark.py"
docker cp "$SCRIPTS_DIR/diagnose_ik_fk.py" \
    "$CONTAINER:$WORKSPACE/scripts/diagnose_ik_fk.py"

echo "Files deployed to $WORKSPACE/scripts/"

MODE="${1:-diagnose}"

if [ "$MODE" = "diagnose" ]; then
    echo ""
    echo "=== Running IK/FK Diagnostic ==="
    docker exec -it "$CONTAINER" bash -c \
        "/isaac-sim/python.sh $WORKSPACE/scripts/diagnose_ik_fk.py"

elif [ "$MODE" = "benchmark" ]; then
    echo ""
    echo "=== Running Sorting Benchmark ==="
    echo "This requires Isaac Sim to be running in the container."
    docker exec -it "$CONTAINER" bash -c \
        "/isaac-sim/python.sh $WORKSPACE/scripts/run_sorting_benchmark.py \
            --benchmark.task_name warehouse_g2 \
            --benchmark.sub_task_name sorting_packages \
            --benchmark.model_arc pi \
            --benchmark.num_episode 1 \
            --app.headless true"

elif [ "$MODE" = "diagnose-only" ]; then
    # Run diagnostic without Isaac Sim python (plain python3)
    echo ""
    echo "=== Running Standalone Diagnostic (no Isaac Sim) ==="
    docker exec -it "$CONTAINER" bash -c \
        "cd $WORKSPACE && python3 scripts/diagnose_ik_fk.py"

else
    echo "Usage: $0 [diagnose|benchmark|diagnose-only]"
    echo "  diagnose      - Run IK/FK diagnostic via /isaac-sim/python.sh"
    echo "  benchmark     - Run full sorting benchmark"
    echo "  diagnose-only - Run diagnostic with plain python3"
    exit 1
fi
