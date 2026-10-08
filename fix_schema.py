"""Normalise a run_grid CSV with mixed old/new schemas to the current FIELDS. usage: python fix_schema.py grid.csv"""
import sys, csv
from run_grid import FIELDS
OLD = [f for f in FIELDS if f not in ("q", "belief", "tau", "planner")]
OLD2 = [f for f in FIELDS if f not in ("tau", "planner")]
OLD3 = [f for f in FIELDS if f not in ("planner", "L")]
OLD4 = [f for f in FIELDS if f not in ("L","divert","n_diverts")]
OLD5 = [f for f in FIELDS if f not in ("divert","n_diverts")]
p = sys.argv[1]
out = []; bad = 0; hdr = None
with open(p, newline="") as f:
    for row in csv.reader(f):
        if not row: continue
        if row[0] == "layout":
            hdr = row; continue
        if len(row) == len(FIELDS) and (hdr is None or hdr == FIELDS): d = dict(zip(FIELDS, row))
        elif hdr is not None and "sim_version" in hdr and len(row) == len(hdr):
            d = {k: "" for k in FIELDS}; d.update({k: v for k, v in zip(hdr, row) if k in d})   # newer columns left blank
        elif len(row) == len(OLD5):
            d = dict(zip(OLD5, row)); d["divert"] = "0"; d["n_diverts"] = "0"
        elif len(row) == len(OLD4):
            d = dict(zip(OLD4, row)); d["L"] = "12600.0"; d["divert"] = "0"; d["n_diverts"] = "0"
        elif len(row) == len(OLD3):
            d = dict(zip(OLD3, row)); d["planner"] = "sa"; d["L"] = "12600.0"
        elif len(row) == len(OLD2):
            d = dict(zip(OLD2, row)); d["tau"] = "0.0"; d["planner"] = "sa"; d["L"] = "12600.0"
        elif len(row) == len(OLD):
            d = dict(zip(OLD, row)); d["q"] = "0.0"; d["belief"] = "prior"; d["tau"] = "0.0"; d["planner"] = "sa"; d["L"] = "12600.0"; d["L"] = "12600.0"
        else: print("malformed row of length", len(row)); bad += 1; continue
        out.append(d)
if bad:
    sys.exit(f"{p}: {bad} row(s) could not be mapped to the current schema; file left unchanged")
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(out)
print(f"rewrote {len(out)} rows with current schema")