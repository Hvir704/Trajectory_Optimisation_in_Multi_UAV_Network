# Baseline registrations (published 20 Oct 2026)

These documents fixed the published-method ports, their one pilot choice and their evaluation rules before the
held-out runs. They were kept on the authors' machines and are published here after the fact. **Their dates are
file dates, not public timestamps**; the only third-party timestamps are the git commits noted below.

| file | what it registers | file date (IST) |
|---|---|---|
| `handover_2026-10-02/planners/GZZ_REGISTRATION.md` | Gao et al. port: pilot dwell rule, held-out protocol, primary criterion | 2 Oct 00:57 |
| `handover_2026-10-02/planners/ASGHAR_REGISTRATION.md` | Asghar et al. port: same | 2 Oct 01:09 |
| `handover_2026-10-02/drl/DRL_REGISTRATION.md` | DRL-UCS(AoI_th) port: slot 5 s, collection radius 50 m, scope, criterion | 2 Oct 01:09 |
| `handover_2026-10-02/SHA256SUMS.txt`, `HANDOVER.md` | hashes of the handed-over code | 2 Oct 01:14 |
| `freeze_2026-10-05/BASELINE_FREEZE_2026-10-05.md` | pilot outcome (dwell mode `rev` for both), DRL scope, code hashes | 5 Oct 05:17 |
| `freeze_2026-10-05/COMMANDS.md`, `evidence/` | commands and the slotted-world equivalence pilot | 5 Oct 05:17 |
| `LIU_EXTENSION_REGISTRATION.md` | Liu et al. extension to the 216 deployments, written 7 Oct before the run | 7-8 Oct |
| `LIU_RYZEN_RUN_INVALID.md` | why the first full Liu run (the annealer under Liu's name) was discarded | 7-8 Oct |

## Checked on 20 Oct

- Every file listed in `handover_2026-10-02/SHA256SUMS.txt` matches its hash.
- `gzz_planner.py` and `asghar_planner.py` are byte-identical in the 2 Oct handover, the 5 Oct freeze, the
  first commit that contains them (67becf2, 7 Oct, together with their held-out results) and the current tree
  (sha256 64db2358... and fa534d61...). The held-out results were therefore produced by the hashed code.
- `dyn_slot_env.py` and `slot_gate.py` (the time-slotted world used only for the DRL attempt and its comparators)
  changed after 2 Oct; the 5 Oct freeze records the two fixes and the new hashes.
- The Liu port was fingerprinted (`check_liu_code.py`) against commit dce4df2 of the reference repository
  (github.com/yugsavalia/BMP-, 7 Oct 02:58 IST); the valid Liu grid ran after that.
- DRL: the registered radius was 50 m. The 200 m and 500 m runs (5 Oct, pilot seed 13) were made after the
  50 m runs failed, so they depart from the registration; the paper says so. No deployment of the registered
  DRL scope was completed, so the registered comparison has no data. Results and logs: `results/drl/`.
