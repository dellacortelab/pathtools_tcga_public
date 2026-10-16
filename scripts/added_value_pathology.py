#!/usr/bin/env python3
"""Exploratory: does any AI-derived quantity add to the clinical grade group for
ADVERSE PATHOLOGY at prostatectomy (extraprostatic disease pT3+, nodal disease N1)?

These outcomes have many more events than recurrence, so the test has real power.
Same candidate family as added_value.py; logistic regression, likelihood-ratio test
of the candidate added to clinical grade group, Benjamini-Hochberg adjusted, and the
optimism-corrected change in AUC. Writes results/added_value_pathology.json.
"""
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from added_value import CANDIDATES, bh
from rules import load_rules

warnings.filterwarnings("ignore")
B = 500


class Logit:
    """Unpenalised logistic regression with log-likelihood and Wald intervals."""

    def __init__(self, y, X):
        self.m = LogisticRegression(penalty=None, max_iter=2000, tol=1e-8).fit(X, y)
        p = np.clip(self.m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        self.llf = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        Xd = np.column_stack([np.ones(len(y)), X])
        cov = np.linalg.inv(Xd.T @ (Xd * (p * (1 - p))[:, None]))
        self.beta, self.se = float(self.m.coef_[0][-1]), float(np.sqrt(cov[-1, -1]))

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


def logit(y, X):
    return Logit(y, X)


def auc_gain(y, base, full, rng):
    def gain(yt, Bt, Ft, ye, Be, Fe):
        return roc_auc_score(ye, logit(yt, Ft).predict(Fe)) - roc_auc_score(ye, logit(yt, Bt).predict(Be))
    app = gain(y, base, full, y, base, full)
    opt, bs = [], []
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        a = gain(y[i], base[i], full[i], y[i], base[i], full[i])
        opt.append(a - gain(y[i], base[i], full[i], y, base, full)); bs.append(a)
    return float(app), float(app - np.mean(opt)), [float(v) for v in np.percentile(bs, [2.5, 97.5])]


def main():
    rng = np.random.default_rng(20261011)
    df = load_rules().dropna(subset=["clin_gg"]).reset_index(drop=True)
    df["crib_share"] = (df["cribriform_area_mm2"] / df["tumor_area_mm2"].replace(0, np.nan)).fillna(0).clip(0, 1)
    known_t = df["stage_t"].astype(str).str.contains(r"T\d")
    known_n = df["stage_n"].astype(str).str.upper().isin(["N0", "N1"]) if "stage_n" in df else None
    import pandas as pd
    from common import DATA
    base_tab = pd.read_csv(os.path.join(DATA, "clinical_baseline.csv"))[["barcode", "PATH_N_STAGE"]]
    df = df.merge(base_tab, on="barcode", how="left")
    df["n1"] = (df["PATH_N_STAGE"] == "N1").astype(int)
    outcomes = {"pT3 or higher": (df[known_t], "pt3plus"),
                "N1 (among patients with assessed nodes)": (df[df["PATH_N_STAGE"].isin(["N0", "N1"])], "n1")}
    res = {"B": B}
    for oname, (d, ycol) in outcomes.items():
        d = d.reset_index(drop=True)
        y = d[ycol].astype(int).values
        base = d[["clin_gg"]].astype(float).values
        m0 = logit(y, base)
        out = {"n": int(len(d)), "events": int(y.sum()), "auc_clinical": float(roc_auc_score(y, d["clin_gg"])), "tests": {}}
        for name, (col, tr) in CANDIDATES.items():
            x = np.log1p(d[col].astype(float)) if tr == "log1p" else d[col].astype(float)
            x = x.values if (col.startswith("ai_gg") or x.nunique() <= 2) else ((x - x.mean()) / x.std()).values
            full = np.column_stack([base, x])
            m1 = logit(y, full)
            ci = np.exp([m1.beta - 1.96 * m1.se, m1.beta + 1.96 * m1.se])
            app, corr, bci = auc_gain(y, base, full, rng)
            out["tests"][name] = dict(odds_ratio=[float(np.exp(m1.beta)), float(ci[0]), float(ci[1])],
                                      lr_p=float(chi2.sf(2 * (m1.llf - m0.llf), 1)),
                                      auc_gain_apparent=app, auc_gain_corrected=corr, auc_gain_ci=bci)
        names = list(out["tests"])
        for n_, a in zip(names, bh([out["tests"][k]["lr_p"] for k in names])):
            out["tests"][n_]["lr_p_bh"] = a
        res[oname] = out
        print(f"\n=== {oname}: n={out['n']} events={out['events']}  AUC clinical GG alone = {out['auc_clinical']:.3f}")
        print(f"{'candidate (per SD / binary), adjusted for clinical GG':52s} {'OR (95% CI)':20s} {'LR p':>9s} {'BH':>8s}  dAUC corrected (95% CI apparent)")
        for k, r in sorted(out["tests"].items(), key=lambda kv: kv[1]["lr_p"]):
            print(f"{k:52s} {r['odds_ratio'][0]:5.2f} ({r['odds_ratio'][1]:.2f}-{r['odds_ratio'][2]:.2f})   {r['lr_p']:9.2g} {r['lr_p_bh']:8.2g}  {r['auc_gain_corrected']:+.3f} ({r['auc_gain_ci'][0]:+.3f} to {r['auc_gain_ci'][1]:+.3f})")
    from common import RESULTS
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "added_value_pathology.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
