# Raw results of the registered held-out tests of theory v3

| file | test | registered with |
|---|---|---|
| `heldout_v3.csv` | test 1 (seeds 13-20; SA, tour, C4) | predictions sha256 4c77e53e... (project log only) |
| `heldout_v3_h24.csv` | test 1, H5 (24 h horizon) | predictions sha256 eb955eb0... (project log only) |
| `heldout2_v3.csv` | test 2 (seeds 21-40; SA, tour, C4, cluster patrol) | manifest 9497b1f4..., commit abac195 |
| `heldout3_v3.csv` | test 3 (seeds 41, 43-49; SA, tour) | manifest 275e503a..., commit 93a3f6e |
| `heldout4_v3.csv` | test 4 (seeds 50-57; SA, tour, C4, cluster patrol) | manifest 4c09326f..., commit 48c72ce |

The `.txt` files are each test's evaluation output as produced on the Ryzen (test 1: the registered evaluation and the
post-registration analyses; tests 2-4: `heldoutN_v3.py evaluate`). Every number was recomputed independently
(`verify_heldout3.py`, `verify_heldout4.py`, and two external reviews: THEORY_V3.md header).
