#!/bin/bash
# reproduce the primary grid + L sweep under an age-blind patrol baseline. One writer per CSV.
PL=$1; OUT=$2
python3 run_grid.py $OUT --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 1-12 --procs ${3:-12} --planner $PL
python3 run_grid.py $OUT --layouts paper --M 400 --Emax 1.5e6 --seeds 1-12 --procs ${3:-12} --planner $PL
python3 run_grid.py $OUT --layouts paper --M 100 --Emax 6e6 --seeds 1-12 --procs ${3:-12} --planner $PL
for L in 8000 10000 16000 20000; do
  python3 run_grid.py $OUT --layouts paper --M 100 --Emax 1.5e6 --seeds 1-8 --procs ${3:-12} --L $L --planner $PL
done
echo DONE
