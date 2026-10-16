#!/usr/bin/env python3
"""Does any AI-derived quantity add prognostic information to the CLINICAL grade group?

Exploratory. Every candidate tested is listed and written to results/added_value.json, with
nominal and multiplicity-adjusted p values, so nothing is reported selectively.

For each candidate x and endpoint (DFI, PFI):
  lr_linear   LR test of x added to a Cox model with clinical GG (linear)
  lr_strat    LR test of x in a Cox model STRATIFIED by clinical GG (no assumption
              about the shape of the clinical-grade effect)
  lr_pt       LR test of x added to clinical GG + pT stage
  dc          change in c-index, clinical GG + x versus clinical GG alone,
              optimism-corrected by bootstrap, with a paired bootstrap CI
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2

from rules import load_rules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RESULTS  # noqa: E402

warnings.filterwarnings("ignore")
B = int(os.environ.get("B", "300"))
T, EV = "dfs_months", "recurred"

CANDIDATES = {  # name: (column, transform)
    "AI grade group, area-ranked": ("ai_gg_area", None),
    "AI grade group, ISUP rule": ("ai_gg_isup", None),
    "AI % Gleason 4+5 of tumour": ("pct_p4p5", None),
    "AI % pattern 4 of tumour": ("pct_p4", None),
    "AI % pattern 5 of tumour": ("pct_p5", None),
    "AI high-grade area / tissue": ("high_grade_frac", None),
    "AI tumour fraction of tissue": ("tumor_fraction", None),
    "AI tumour area (log mm2)": ("log_tumor_area", None),
    "AI largest focus (log mm2)": ("log_largest_focus", None),
    "AI number of foci (log)": ("n_foci", "log1p"),
    "AI cribriform area (log mm2)": ("cribriform_area_mm2", "log1p"),
    "AI cribriform present": ("cribriform_present", None),
    "AI cribriform share of tumour area": ("crib_share", None),
    "AI perineural invasion present": ("pni_present", None),
    "AI perineural invasion count (log)": ("pni_count", "log1p"),
    "AI mean gland confidence": ("mean_gland_conf", None),
}


def fit(d, cols, strata=None):
    return CoxPHFitter().fit(d[cols + [T, EV] + ([strata] if strata else [])], T, EV, strata=[strata] if strata else None)


def lr(full, red, dof=1):
    return float(chi2.sf(max(0.0, 2 * (full.log_likelihood_ - red.log_likelihood_)), dof))


def cidx(d, f, cols):
    return concordance_index(d[T], -f.predict_log_partial_hazard(d[cols]).values.ravel(), d[EV])


def delta_c(d, x, rng):
    base, full = ["clin_gg"], ["clin_gg", x]
    app = cidx(d, fit(d, full), full) - cidx(d, fit(d, base), base)
    opt, boots = [], []
    n = len(d)
    for _ in range(B):
        s = d.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        if s[EV].sum() < 5 or s[x].nunique() < 2:
            continue
        try:
            fb, ff = fit(s, base), fit(s, full)
        except Exception:
            continue
        in_boot = cidx(s, ff, full) - cidx(s, fb, base)
        in_orig = cidx(d, ff, full) - cidx(d, fb, base)
        opt.append(in_boot - in_orig); boots.append(in_boot)
    return dict(apparent=float(app), corrected=float(app - np.mean(opt)),
                ci=[float(v) for v in np.percentile(boots, [2.5, 97.5])])


def bh(ps):
    ps = np.asarray(ps, float); n = len(ps); order = np.argsort(ps)
    adj = np.empty(n); prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, ps[i] * n / (n - rank)); adj[i] = prev
    return adj.tolist()


def run(csv, rng):
    df = load_rules(os.path.join(DATA, csv)).dropna(subset=["clin_gg"])
    df = df[df[T] > 0].reset_index(drop=True)
    df["crib_share"] = (df["cribriform_area_mm2"] / df["tumor_area_mm2"].replace(0, np.nan)).fillna(0).clip(0, 1)
    out = {"n": int(len(df)), "events": int(df[EV].sum()), "tests": {}}
    for name, (col, tr) in CANDIDATES.items():
        d = df.copy()
        x = np.log1p(d[col].astype(float)) if tr == "log1p" else d[col].astype(float)
        # grade groups stay per group and binary flags per presence; everything else per SD
        d["x"] = x if (col.startswith("ai_gg") or x.nunique() <= 2) else (x - x.mean()) / x.std()
        r = {}
        f0, f1 = fit(d, ["clin_gg"]), fit(d, ["clin_gg", "x"])
        s = f1.summary.loc["x"]
        r["hr"] = [float(np.exp(s["coef"])), float(np.exp(s["coef lower 95%"])), float(np.exp(s["coef upper 95%"]))]
        r["lr_linear"] = lr(f1, f0)
        try:
            fs = fit(d, ["x"], strata="clin_gg")
            r["lr_strat"] = float(fs.log_likelihood_ratio_test().p_value)
            r["hr_strat"] = float(np.exp(fs.summary.loc["x", "coef"]))
        except Exception:
            r["lr_strat"] = float("nan")
        r["lr_pt"] = lr(fit(d, ["clin_gg", "pt_stage_num", "x"]), fit(d, ["clin_gg", "pt_stage_num"]))
        r["dc"] = delta_c(d, "x", rng)
        out["tests"][name] = r
    for key in ("lr_linear", "lr_strat", "lr_pt"):
        names = list(out["tests"])
        adj = bh([out["tests"][n][key] for n in names])
        for n, a in zip(names, adj):
            out["tests"][n][key + "_bh"] = a
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(20261011)
    res = {"B": B, "n_candidates": len(CANDIDATES),
           "DFI": run("features.csv", rng), "PFI": run("features_pfs.csv", rng)}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "added_value.json"), "w"), indent=1)
    for ep in ("DFI", "PFI"):
        e = res[ep]
        print(f"\n=== {ep}: n={e['n']} events={e['events']}  ({len(CANDIDATES)} candidates; BH = Benjamini-Hochberg)")
        print(f"{'candidate (per SD / binary)':38s} {'HR|clinGG (95% CI)':22s} {'LR p':>7s} {'BH':>6s} {'strat p':>8s} {'BH':>6s} {'+pT p':>7s} {'BH':>6s}  {'dC corr (95% CI apparent)':s}")
        for n, r in sorted(e["tests"].items(), key=lambda kv: kv[1]["lr_linear"]):
            print(f"{n:38s} {r['hr'][0]:5.2f} ({r['hr'][1]:.2f}-{r['hr'][2]:.2f})      {r['lr_linear']:7.3f} {r['lr_linear_bh']:6.2f} {r['lr_strat']:8.3f} {r['lr_strat_bh']:6.2f} {r['lr_pt']:7.3f} {r['lr_pt_bh']:6.2f}  {r['dc']['corrected']:+.3f} ({r['dc']['ci'][0]:+.3f} to {r['dc']['ci'][1]:+.3f})")
