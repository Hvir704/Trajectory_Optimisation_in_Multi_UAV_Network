# Design-time scripts for theory v3.3 / v3.4 (exploratory, not registered)

These produced the seen-data tables of THEORY_V3.md sections 10.1 and 11. Run them from the repository root after
copying them there, with two environment variables set:

- `V3_ASSEMBLED`: the folder with the assembled v2.2.1 grid (`results_v2/report/assembled`);
- `V3_RESULTS`: the folder with the held-out CSVs (`theory_v3_results`).

Run `eval_v33.py` first: it builds the tour cache (`tour_cache_v3.pkl`) the others reuse.
