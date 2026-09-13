"""Prepend the CSV header to a headerless run_grid output. usage: python fix_header.py w4.csv"""
import sys
from run_grid import FIELDS
p = sys.argv[1]; body = open(p).read()
if not body.startswith("layout,"):
    open(p, "w", newline="").write(",".join(FIELDS) + "\r\n" + body)
    print("header added")
else:
    print("already has header")