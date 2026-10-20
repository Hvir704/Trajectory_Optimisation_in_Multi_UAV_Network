# Liu extension: the Ryzen run (3,613 + 3 jobs) is NOT usable -- it was SA, not Liu

## ROOT CAUSE (confirmed)
The Ryzen folder has no liu_mpga.py, and its run_grid.py has no liu_mpga branch: an unknown planner name falls through
to `else: build_sa_planner(...)`. Every "Liu" row is an SA run. Proof: core M=50 1.5 MJ s3 K2 "Liu" = 299153.45190639276,
i9 SA = 299153.4519063928; over 3,099 shared jobs 76% agree with i9 SA within 1% (rest = SA cross-machine noise).
The earlier "i5 vs Ryzen" comparison below therefore compared Liu with SA, and the "code drift" reading is superseded.
Fault: the paper chat adapted the run instructions to the user's own repo without checking it contained the Liu
planner, and run_liu_grid.py did not verify the planner. run_liu_grid.py now aborts unless run_grid has a liu_mpga
branch and liu_mpga is importable (tested: passes in the reference folder, aborts in a folder like the Ryzen's).


## What the data show
- File integrity is fine: all 3,613 jobs present once, no malformed or duplicated rows; one extra header line at the
  top (runner started against a header-only file) -- harmless.
- But the Ryzen's Liu results disagree with the i5 machine's on the 404 shared M=50 jobs: only 57 bit-identical,
  median |diff| 4%, 90th percentile 63%, max 118%; Ryzen WORSE in 265, better in 82 (not symmetric noise).
  Liu's optimal fleet size changes in 9/36 deployments; optimal J moves by a median +5.7% (up to +79%).
- The paper-chat sandbox (Linux, different hardware) running the BMP repo code reproduces the i5 values on every job
  tried (core s3 K2: 1.37183e5 on both; Ryzen 2.99e5). So it is a CODE difference, not a platform effect.
- The same BMP code reproduces the canonical cluster-patrol and C4 values (eval_all_v2.csv) bit for bit, so the BMP
  code IS the reference simulator behind the paper's tables. The Ryzen copy is not.
- On core s3 K2 the Ryzen Liu result is 2.2x worse than the cluster patrol, i.e. worse than the GA's own starting
  partition -- elitism should make that impossible. Prime suspect: a different cyclic_sched.py (Ctx surrogate,
  cut_sortie) or liu_mpga.py in the Ryzen folder.
- Correction to an earlier statement: "the MPGA is not deterministic across machines" was wrong as a reading of the
  i5-vs-Ryzen gap; with the SAME code, two machines agree exactly on every job checked.

## What to do on the Ryzen
1. In the folder used for the run:  python check_liu_code.py   (expect FAIL; send the output -- it names the
   file that differs).
2. Rerun from a clean copy of the reference code (public repo, no credentials needed):
       git clone https://github.com/yugsavalia/BMP- bmp_ref
       cd bmp_ref
       (copy in: check_liu_code.py -> repo root; run_liu_grid.py -> baselines\liu2022\; liu_jobs_all.csv -> repo root)
       python check_liu_code.py                                   # must print PASS before anything else
       python baselines\liu2022\run_liu_grid.py liu_jobs_all.csv liu_grid_ref.csv --procs 20
       $env:PYTHONPATH = "sim;experiments;baselines\liu2022"
       python experiments\fill_tail.py liu_fill_ref.csv --inputs liu_grid_ref.csv --planner liu_mpga --rule15 --procs 20
   Use the NEW file names (liu_grid_ref.csv, liu_fill_ref.csv); do not append to the old files. ~3 h.
3. Upload liu_grid_ref.csv and liu_fill_ref.csv. The paper chat re-checks a sample against the sandbox before the
   registered analysis.

## Unaffected
The paper's existing results (SA, C4, cluster patrol, Gao, Asghar): the reference code reproduces C4 and the
cluster patrol exactly. Nothing from the Ryzen Liu run has entered the paper.
