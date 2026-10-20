# Reproducing the paper

All commands run from the repository root with Python 3.11+ and the packages in `requirements.txt`
(numpy, pandas, scipy, matplotlib, ortools). Section, table and figure numbers refer to `paper_v22/main.tex`
as compiled (Table II = certification ... Table XVII = fitted exponent).

## 1. Everything in Sections VI-VII and the appendices, from the recorded runs (no simulation)

The run files (`results_v2/`, about 30 MB) are attached to the repository's tagged release. Unpack them into
`results_v2/` and run

```
python reproduce_v2.py results_v2
```

It checks the simulator version of every file, assembles them, reports every sweep that does not meet the
extension rule, and writes one text file per step to `results_v2/report/` (about 25 minutes on two cores). Each
step's label names the table, figure or section it feeds; `summary` at the end lists any failed step.
`python reproduce_v2.py results_v2 --only <step>` reruns one step.

## 2. Side results that come from their own scripts

| result | command |
|---|---|
| Table II (per-decision certification, M=14, 15) | `python certify_sortie.py 14 1-6 3e5 2 39` and `python certify_sortie.py 15 1-6 3e5 2 39` |
| CP-SAT against Held-Karp, M=12 (Sec. VI-A) | `python milp_probe.py check 12 1-1 3e5 2 16` |
| CP-SAT per-decision probe, M=40 (Sec. VI-A) | `python milp_probe.py time 40 1-6 1.5e6 4 3 120 8` |
| Decision cost (Sec. VI-C) | `python latency.py` (timing; machine-dependent) |
| Fig. 1 | `python plot_snapshot_pair.py 100 1.5e6 4 5 1 6 fig1_reach.pdf` |
| eta Monte Carlo (Sec. V-E) | `python eta_mc.py 2000` (also a step of `reproduce_v2.py`) |

The simulation runs themselves are listed, with their exact commands, in `run_all_ryzen.ps1`.

## 3. The served-set rule and its four registered held-out tests (Sec. V-D, VII-E)

| item | files |
|---|---|
| rule | `theory_v3.py` (v3, v3.3), `theory_v34.py` (crowding-corrected hop), `localcheck.py` (local check) |
| test N | `heldout_v3.py` (test 1), `heldout2_v3.py`, `heldout3_v3.py`, `heldout4_v3.py`; predictions `heldout*_predictions.csv` |
| registration | `registrations/` (manifests); commits `abac195`, `93a3f6e`, `48c72ce` |
| raw runs and evaluations | `theory_v3_results/` |
| independent re-evaluation | `python verify_heldout3.py theory_v3_results/heldout3_v3.csv heldout3_predictions.csv` (and `verify_heldout4.py`) |

`python heldoutN_v3.py predict` regenerates a prediction file. On Linux the CSV writer emits CRLF line endings;
the content is identical to the committed file after line endings are normalised, and the manifests hash
LF-normalised bytes, so `manifest_sha()` reproduces the registered hashes. `python heldoutN_v3.py evaluate`
prints the registered verdicts.

## 4. Version

Every run file carries `sim_version = v2.2`. The paper as of the theory-v3 draft is tag `paper-v3`; the earlier
v2.2.1 paper is tag `paper-v2.2.1`.
