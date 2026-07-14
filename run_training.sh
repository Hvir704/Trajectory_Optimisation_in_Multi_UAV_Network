#!/usr/bin/env bash
# ============================================================================
# run_training.sh — Multi-UAV training sweep (SPLIT battery, E_max/K per UAV)
# M values : 50 60 80 100 120 150 200
# K values : 1 2 3 4  (plus K=6 at M=100 for saturation curve)
# Battery  : SPLIT is default — no extra flag needed
# ============================================================================
# Usage:
#   bash run_training.sh          # full sweep
#   bash run_training.sh quick    # smoke pass (confirms wiring, ~5 min)
#
# Prerequisites:
#   - uav_aoi_solver.py and multi_uav_solver.py in the same folder
#   - pip install torch numpy matplotlib
#   - Run from that same folder
# ============================================================================
set -e

MODE="${1:-full}"

if [ "$MODE" = "quick" ]; then
    echo "### QUICK smoke pass ###"
    python multi_uav_solver.py --quick --M 50 --K 1 2 3 --epochs 20 --instances 30
    echo "### Smoke pass done — if no errors, run: bash run_training.sh ###"
    exit 0
fi

EPOCHS=300
INSTANCES=200
SEED=42

echo "============================================================"
echo "  Multi-UAV sweep  |  split battery  |  epochs=$EPOCHS"
echo "  M = 50 60 80 100 120 150 200   K = 1 2 3 4"
echo "  M=100 also gets K=6 for saturation curve"
echo "============================================================"

python multi_uav_solver.py --M 50  --K 1 2 3 4   --epochs $EPOCHS --instances $INSTANCES --seed $SEED
python multi_uav_solver.py --M 60  --K 1 2 3 4   --epochs $EPOCHS --instances $INSTANCES --seed $SEED
python multi_uav_solver.py --M 80  --K 1 2 3 4   --epochs $EPOCHS --instances $INSTANCES --seed $SEED
python multi_uav_solver.py --M 100 --K 1 2 3 4 6 --epochs $EPOCHS --instances $INSTANCES --seed $SEED
python multi_uav_solver.py --M 120 --K 1 2 3 4   --epochs $EPOCHS --instances $INSTANCES --seed $SEED
python multi_uav_solver.py --M 150 --K 1 2 3 4   --epochs $EPOCHS --instances $INSTANCES --seed $SEED
python multi_uav_solver.py --M 200 --K 1 2 3 4   --epochs $EPOCHS --instances $INSTANCES --seed $SEED

echo "============================================================"
echo "  Done."
echo "  Models  -> models_multi_uav/fleet_M{M}_K{K}_split.pt"
echo "  Figures -> results/fig_multiuav_objective_vs_K_M{M}.png"
echo "             results/fig_multiuav_trajectory_M{M}_K{K}.png"
echo "============================================================"
