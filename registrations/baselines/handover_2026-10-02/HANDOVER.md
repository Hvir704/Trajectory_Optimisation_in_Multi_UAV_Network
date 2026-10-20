# Baseline handover (2 Oct 2026)

Contents
- `planners/` — CPU-side baselines for run_grid (copy into your project folder, next to dyn_env.py):
  `gzz_planner.py` (Gao-Zhu-Zhai, TWC 2023), `asghar_planner.py` (Asghar-Sundaram-Smith, T-RO 2025),
  `run_grid.py` (registers gzz_full/gzz_rev/asghar_full/asghar_rev; CRLF kept), `dyn_slot_env.py` + `slot_gate.py`
  (our world in 5 s slots; flies any launch-time planner there), `GZZ_REGISTRATION.md`, `ASGHAR_REGISTRATION.md`.
- `drl/` — DRL-UCS(AoI_th) (ToN 2024) on the RTX 5080: `drl_ucs_modernisation_and_envdyn.patch`,
  `make_drl_configs.py`, `configs/`, `DRL_REGISTRATION.md`.

New dependency for the planners: `pip install -r planners/requirements.txt` (adds ortools/CP-SAT for asghar_planner).

---------------------------------------------------------------------------------------------------
## A. GPU machine (RTX 5080) — DRL baseline

Use Linux or WSL2 (their env manager uses subprocess/zmq and Ray multi-process; native Windows is not supported by
this stack). The 5080 is Blackwell (sm_120): it needs a PyTorch build with CUDA 12.8 or newer. The authors' pinned
stack (Python 3.8, PyTorch 1.11, CUDA 11.6) cannot use it; the patch ports their code to current Python/PyTorch/Ray.

### A1. Environment (once) -- venv + requirements.txt
```bash
python3.12 -m venv ~/venvs/drlucs && source ~/venvs/drlucs/bin/activate     # Python 3.10-3.12 all fine
pip install --upgrade pip
pip install -r /path/to/drl/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
git clone https://github.com/BIT-MCS/DRL-UCS-AoI-Threshold.git && cd DRL-UCS-AoI-Threshold
patch -p1 < /path/to/drl/drl_ucs_modernisation_and_envdyn.patch
touch DRL_UCS_AoI/adept/env/env_dyn/__init__.py          # empty file, not carried by the patch
pip install -e DRL_UCS_AoI --no-deps                       # their setup.py pins tensorflow etc.; --no-deps keeps the venv clean
export PYTHONPATH=$PWD/DRL_UCS_AoI                        # add to ~/drlucs.env and source it in every shell
```
If `torch.cuda.is_available()` is False: `nvidia-smi` must show driver >= 580 (WSL2: update the Windows driver);
otherwise use the cu128 line in requirements.txt.
Tested here (CPU only): Python 3.12, torch 2.14, Ray 2.58, gym 0.26.2, NumPy 2.4. Train -> checkpoint -> clean exit ->
evaluate.py ran end to end. Not tested here: anything on a GPU (no GPU in my sandbox).

What the patch changes (each line marked [modernisation] or [fix] in the code): NumPy-2 dtype map; Ray 2.x
(removed `ray.util.sgd`, `ray.services`, `local_mode`); hard-coded "cuda" devices -> the actual device;
CUDA_VISIBLE_DEVICES no longer forced ("" = leave as is); stray unused imports (tkinter, h11, matplotlib);
training never exited after the last checkpoint (queue/thread deadlock) — fixed; evaluator strict weight loading
vs its own key filter — fixed; plus the new environment `EnvDyn-v0` (adept/env/env_dyn: our simulator vendored).

### A2. Smoke test on the GPU (5 min)
```bash
cd DRL_UCS_AoI/adept/scripts
python actorlearner.py --config /path/to/drl/configs/smoke_cpu.json --learner-gpu-alloc 0.5 --worker-gpu-alloc 0.1
```
(or edit the two *_gpu_alloc keys in the JSON; set "logdir" to a real path). Expect "0 stopped training" and
checkpoints in <logdir>/EnvDyn-v0/<tag>/<step>/model_<step>.pth. Then:
```bash
python evaluate.py --logdir <logdir>/EnvDyn-v0/<tag> --nb-episode 1 --gpu-id 0
```
The log ends with `e_weighted_aoi : <J>` (= J_timeavg, true world).

### A3. Pipeline check: reproduce one number from their paper (Beijing, U = 3)
Their own environment and settings, on the modernised stack. In `adept/config/bj.json` set `"cuda_visible_device": ""`,
`"logdir"` to your path, and `"nb_learn_batch"` <= `"nb_workers"` (as released, 8 > 4 fails the launcher's own
assertion). Train (`python actorlearner.py --config ../config/bj.json`) and evaluate. Paper Table/Fig. 4: threshold AoI
kappa = 26.84 timeslots at U = 3 (Beijing). Record the gap; do not tune to close it.
(Their bj.json budget is 1e8 steps; use 1e7 for this check if time is short and say so.)

### A4. Throughput pilot (decides the scope) — about 20 min per config
Set workers/envs to your machine first: nb_workers ~= (physical cores - 2) / 2, nb_env 8 per worker.
```bash
python make_drl_configs.py --out configs/tp --layout paper --M 50 --Emax 1.5e6 --seeds 13 --K 3 --nb-step 3e5 --workers <W> --envs 8 --logdir ~/drl_logs
python make_drl_configs.py --out configs/tp --layout core  --M 200 --Emax 3e6  --seeds 13 --K 8 --nb-step 3e5 --workers <W> --envs 8 --logdir ~/drl_logs
python actorlearner.py --config configs/tp/<file>.json      # each; note LOCAL STEP/S per worker x workers
```
Watch `nvidia-smi` (GPU util) and `htop` (CPU): if CPUs are pegged and the GPU is idle, the environment is the
bottleneck -> fewer envs per worker won't help, more workers (cores) will. Report back: steps/s for both configs,
core count, GPU util. Hours per training run = 1e7 / (steps/s) / 3600.

### A5. Training runs (after the scope is registered in DRL_REGISTRATION.md)
```bash
python make_drl_configs.py --out configs/run --layout paper --M 50 --Emax 1.5e6 --seeds 1 2 3 --K <union K grid> --nb-step 1e7 --workers <W> --envs 8 --logdir ~/drl_logs
for c in configs/run/*.json; do python actorlearner.py --config $c; done      # sequential: one run owns the GPU
for d in ~/drl_logs/EnvDyn-v0/*; do python evaluate.py --logdir $d --nb-episode 1 --gpu-id 0 | tail -20 > $d/eval.txt; done
```
Send back the eval.txt files (or the whole logdir minus checkpoints) and the tensorboard event files (learning curves).

---------------------------------------------------------------------------------------------------
## B. CPU machines (i9 / Ryzen) — classical baselines and slotted comparators
```bash
pip install -r planners/requirements.txt
# gzz and asghar pilots (seeds 13-15), commands and decision rules in GZZ_REGISTRATION.md / ASGHAR_REGISTRATION.md
python run_grid.py gzz_pilot.csv    --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --planner gzz_full    --procs 8
python run_grid.py gzz_pilot.csv    --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --planner gzz_rev     --procs 8
python run_grid.py asghar_pilot.csv --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --planner asghar_full --procs 8
python run_grid.py asghar_pilot.csv --layouts paper ring core --M 50 100 200 --Emax 1.5e6 3e6 --seeds 13-15 --planner asghar_rev  --procs 8
# collection-radius pilot for the DRL port (scripted routes, slotted world vs DynSim), seeds 13-15
python slot_gate.py --cells paper:50:1.5e6:3 paper:50:1.5e6:5 core:100:3e6:6 ring:100:3e6:6 --slots 5 --planners rr_tour greedy cluster_patrol --seeds 13 14 15 --csv gate_pilot.csv
```
After the DRL scope is fixed: comparators in the same slotted world on the chosen deployments, e.g.
`python slot_gate.py --cells paper:50:1.5e6:<K> ... --slots 5 --planners cluster_patrol gzz_<mode> asghar_<mode> cyc --seeds 1 2 3 --csv drl_comparators.csv`
(cyc needs cyclic_sched.py next to it; sa runs too but take SA from the i9 as usual).

Send back: the pilot CSVs + `[gzz]` / `[asghar]` log lines, gate_pilot.csv, and the A4 throughput numbers.
I then fill in the dwell-mode choices, the collection radius and the DRL scope, and freeze the registrations.
