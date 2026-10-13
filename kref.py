"""kref.py -- the reference fleet size of an instance, at which the rule's measured commute inputs are read.

It is the first K of the instance's ORIGINAL sweep (its base run file, before any fill), so the rule's inputs do not
depend on how far a fill later extended that sweep, in either direction. reproduce_v2.py attaches it to every
assembled row as the column K_ref. Without that column (raw files) it falls back to the smallest K present.
"""


def kref(byK):
    for v in byK.values():
        if isinstance(v, dict) and v.get("K_ref") not in (None, ""):
            k = int(float(v["K_ref"]))
            return k if k in byK else min(byK)
        break
    return min(byK)
