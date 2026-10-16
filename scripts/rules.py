"""Grade-group aggregation rules applied to the AI pattern composition.

`area_ranked`  the pre-specified rule: primary = largest
               pattern area, secondary = second largest, however small.
`isup_rp`      ISUP radical-prostatectomy convention with a minor-pattern
               threshold (default 5 % of tumour): a pattern below the threshold
               is not incorporated into the score; if more than two patterns
               clear the threshold the score is primary + highest.
Both take the tumour-normalised shares (p3, p4, p5) and return (primary,
secondary, grade_group).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load  # noqa: E402

MINOR = 0.05


def grade_group(primary, secondary):
    if primary == 0:
        return 0
    s = primary + secondary
    if s <= 6:
        return 1
    if (primary, secondary) == (3, 4):
        return 2
    if (primary, secondary) == (4, 3):
        return 3
    if s == 8:
        return 4
    return 5


def area_ranked(p3, p4, p5):
    a = {3: p3, 4: p4, 5: p5}
    r = sorted([g for g in a if a[g] > 0], key=lambda g: -a[g])
    if not r:
        return 0, 0, 0
    pr, se = (r[0], r[0]) if len(r) == 1 else (r[0], r[1])
    return pr, se, grade_group(pr, se)


def isup_rp(p3, p4, p5, minor=MINOR):
    a = {3: p3, 4: p4, 5: p5}
    tot = sum(a.values())
    if tot <= 0:
        return 0, 0, 0
    kept = [g for g in a if a[g] / tot >= minor]
    if not kept:  # nothing clears the threshold: fall back to the dominant pattern
        kept = [max(a, key=lambda g: a[g])]
    pr = max(kept, key=lambda g: a[g])
    others = [g for g in kept if g != pr]
    if not others:
        se = pr
    elif len(others) == 1:
        se = others[0]
    else:  # three patterns above threshold -> primary + highest
        se = max(others)
    return pr, se, grade_group(pr, se)


def add_rules(df, minor=MINOR):
    """Add ai_gg_area (== submitted grade) and ai_gg_isup columns."""
    sh = df[["pct_p3", "pct_p4", "pct_p5"]].astype(float).values
    df = df.copy()
    df["ai_gg_area"] = [area_ranked(*r)[2] for r in sh]
    df["ai_gg_isup"] = [isup_rp(*r, minor=minor)[2] for r in sh]
    return df


def load_rules(path=None, minor=MINOR):
    df = load(path) if path else load()
    df = add_rules(df, minor)
    assert (df["ai_gg_area"] == df["ai_gg"]).all(), "area-ranked rule must reproduce the submitted grade"
    return df
