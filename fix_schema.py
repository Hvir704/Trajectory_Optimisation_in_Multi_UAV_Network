"""Normalise a run_grid CSV with mixed old/new schemas to the current FIELDS. usage: python fix_schema.py grid.csv"""
import sys, csv
from run_grid import FIELDS
OLD = [f for f in FIELDS if f not in ("q", "belief", "tau", "planner")]
OLD2 = [f for f in FIELDS if f not in ("tau", "planner")]
OLD3 = [f for f in FIELDS if f not in ("planner", "L")]
OLD4 = [f for f in FIELDS if f not in ("L","divert","n_diverts")]
OLD5 = [f for f in FIELDS if f not in ("divert","n_diverts")]
p = sys.argv[1]
out = []
with open(p, newline="") as f:
    for row in csv.reader(f):
        if not row or row[0] == "layout": continue
        if len(row) == len(FIELDS): d = dict(zip(FIELDS, row))
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
        else: print("skipping malformed row of length", len(row)); continue
        out.append(d)
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(out)
print(f"rewrote {len(out)} rows with current schema")