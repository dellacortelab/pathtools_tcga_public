#!/usr/bin/env python3
"""Main analysis: every statistic in the manuscript except the exploratory
added-value analysis, written to results/analysis.json.

  cohort table inputs (Table 1)
  grade-group concordance under both aggregation rules, misclassification at
    clinical cut-points, minor-pattern threshold sweep
  standalone discrimination and Cox models for DFI, PFI and PSA recurrence
    (paired bootstrap), joint models
  grade + pT stage: optimism correction and pT-imputation sensitivity
  calibration
Run:  python analysis.py      (B=200 in the environment for a quick run)
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from rules import add_rules, load_rules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import logrank, DATA, RESULTS  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20261010
B = int(os.environ.get("B", "2000"))
HORIZON = 36.0  # months, for calibration
OUT = os.path.join(RESULTS, "analysis.json")
ENDPOINT_CSV = {"DFI": "features.csv", "PFI": "features_pfs.csv", "BCR": "features_psa.csv"}


def cstat(t, e, score):
    return float(concordance_index(t, -np.asarray(score, float), e))


def ci(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return [float(x) for x in np.percentile(a, [2.5, 97.5])]


def cox_lp(train, test, cols, penalizer=0.0):
    d = train[["dfs_months", "recurred"] + cols]
    f = CoxPHFitter(penalizer=penalizer).fit(d, "dfs_months", "recurred")
    return f, f.predict_log_partial_hazard(test[cols]).values.ravel()


# ---------------------------------------------------------------- concordance
def concordance(d, col, rng):
    c, a = d["clin_gg"].astype(int).values, d[col].astype(int).values
    n = len(d)
    ks = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        ks.append(cohen_kappa_score(c[i], a[i], weights="quadratic"))
    M = np.zeros((5, 5), int)
    for x, y in zip(c, a):
        M[x - 1, y - 1] += 1
    return dict(n=n, kappa=float(cohen_kappa_score(c, a, weights="quadratic")), kappa_ci=ci(ks),
                exact=float((c == a).mean()), within1=float((abs(c - a) <= 1).mean()),
                over=float((a > c).mean()), under=float((a < c).mean()),
                confusion=M.tolist(),
                mean_ai_by_clin=[float(a[c == g].mean()) for g in range(1, 6)],
                sd_ai_by_clin=[float(a[c == g].std(ddof=1)) for g in range(1, 6)])


def thresholds(d, col):
    """Misclassification against clinically used grade cut-points."""
    c, a = d["clin_gg"].astype(int).values, d[col].astype(int).values
    out = {}
    for name, cut in (("GG1_vs_GG2plus", 2), ("GG1-2_vs_GG3plus", 3), ("GG1-3_vs_GG4plus", 4)):
        pos_c, pos_a = c >= cut, a >= cut
        out[name] = dict(
            n_low=int((~pos_c).sum()), low_called_high=int((~pos_c & pos_a).sum()),
            n_high=int(pos_c.sum()), high_called_low=int((pos_c & ~pos_a).sum()),
            sensitivity=float((pos_c & pos_a).sum() / pos_c.sum()),
            specificity=float((~pos_c & ~pos_a).sum() / (~pos_c).sum()))
    out["GG5_called_GG3_or_lower"] = [int(((c == 5) & (a <= 3)).sum()), int((c == 5).sum())]
    out["GG1_called_GG3plus"] = [int(((c == 1) & (a >= 3)).sum()), int((c == 1).sum())]
    return out


# ------------------------------------------------------------- discrimination
def discrimination(d, rng, preds):
    d = d.reset_index(drop=True)
    t, e = d["dfs_months"].values, d["recurred"].values
    pts = {p: cstat(t, e, d[p]) for p in preds}
    boots = {p: [] for p in preds}
    n = len(d)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if e[i].sum() < 2:
            continue
        for p in preds:
            boots[p].append(cstat(t[i], e[i], d[p].values[i]))
    out = {"n": int(n), "events": int(e.sum())}
    ref = np.array(boots["clin_gg"])
    for p in preds:
        b = np.array(boots[p])
        out[p] = dict(c=pts[p], ci=ci(b))
        if p != "clin_gg":
            out[p]["delta_clin_minus"] = float(pts["clin_gg"] - pts[p])
            out[p]["delta_ci"] = ci(ref - b)
    return out


def logrank3(d, col):
    g = np.where(d[col] <= 2, "lo", np.where(d[col] == 3, "mid", "hi"))
    return float(logrank(d["dfs_months"], d["recurred"], g)[1])


def unicox(d, col, std=False, penalizer=0.0):
    x = d[[col, "dfs_months", "recurred"]].dropna().copy()
    if std:
        x[col] = (x[col] - x[col].mean()) / x[col].std()
    f = CoxPHFitter(penalizer=penalizer).fit(x, "dfs_months", "recurred")
    r = f.summary.loc[col]
    return dict(hr=float(np.exp(r["coef"])), lo=float(np.exp(r["coef lower 95%"])),
                hi=float(np.exp(r["coef upper 95%"])), p=float(r["p"]))


def joint(d, a, b):
    """Unpenalised joint Cox of two predictors with bidirectional likelihood-ratio tests."""
    from scipy.stats import chi2
    x = d[[a, b, "dfs_months", "recurred"]].dropna()
    fa = CoxPHFitter().fit(x[[a, "dfs_months", "recurred"]], "dfs_months", "recurred")
    fb = CoxPHFitter().fit(x[[b, "dfs_months", "recurred"]], "dfs_months", "recurred")
    fj = CoxPHFitter().fit(x, "dfs_months", "recurred")
    s = fj.summary
    return {"n": int(len(x)), "events": int(x["recurred"].sum()), "c_joint": float(fj.concordance_index_),
            f"p_{b}_adds_to_{a}": float(chi2.sf(2 * (fj.log_likelihood_ - fa.log_likelihood_), 1)),
            f"p_{a}_adds_to_{b}": float(chi2.sf(2 * (fj.log_likelihood_ - fb.log_likelihood_), 1)),
            "hr": {k: [float(np.exp(s.loc[k, "coef"])), float(np.exp(s.loc[k, "coef lower 95%"])),
                       float(np.exp(s.loc[k, "coef upper 95%"])), float(s.loc[k, "p"])] for k in (a, b)},
            "spearman": float(x[a].corr(x[b], method="spearman"))}


# ------------------------------------------------------ optimism / imputation
def optimism(d, cols, rng, penalizer=0.0, b=500):
    d = d[["dfs_months", "recurred"] + cols].dropna().reset_index(drop=True)
    f, lp = cox_lp(d, d, cols, penalizer)
    app = cstat(d["dfs_months"], d["recurred"], lp)
    opt = []
    n = len(d)
    for _ in range(b):
        s = d.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        if s["recurred"].sum() < 3 or s[cols].nunique().min() < 2:
            continue
        try:
            _, lp_b = cox_lp(s, s, cols, penalizer)
            _, lp_o = cox_lp(s, d, cols, penalizer)
        except Exception:
            continue
        opt.append(cstat(s["dfs_months"], s["recurred"], lp_b) - cstat(d["dfs_months"], d["recurred"], lp_o))
    # repeated stratified 5-fold cross-validation, pooled out-of-fold predictions
    cv = []
    for rep in range(40):
        oof = np.zeros(n)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(d, d["recurred"]):
            _, oof[te] = cox_lp(d.iloc[tr], d.iloc[te], cols, penalizer)
        cv.append(cstat(d["dfs_months"], d["recurred"], oof))
    return dict(n=int(n), apparent=app, optimism=float(np.mean(opt)), corrected=float(app - np.mean(opt)),
                cv_mean=float(np.mean(cv)), cv_sd=float(np.std(cv)))


def pt_sensitivity(df, ai="ai_gg"):
    out = {}
    known = df["stage_t"].astype(str).str.contains(r"T\d")
    variants = {"imputed_pT2 (submitted)": df.assign(pt=df["pt_stage_num"]),
                "complete_case": df[known].assign(pt=df.loc[known, "pt_stage_num"]),
                "imputed_pT3": df.assign(pt=np.where(known, df["pt_stage_num"], 3))}
    for k, d in variants.items():
        x = d[[ai, "pt", "dfs_months", "recurred"]].dropna()
        f = CoxPHFitter().fit(x, "dfs_months", "recurred")
        s = f.summary
        out[k] = dict(n=int(len(x)), events=int(x["recurred"].sum()), c=float(f.concordance_index_),
                      pt_hr=float(np.exp(s.loc["pt", "coef"])), pt_lo=float(np.exp(s.loc["pt", "coef lower 95%"])),
                      pt_hi=float(np.exp(s.loc["pt", "coef upper 95%"])), pt_p=float(s.loc["pt", "p"]),
                      ai_hr=float(np.exp(s.loc[ai, "coef"])), ai_lo=float(np.exp(s.loc[ai, "coef lower 95%"])),
                      ai_hi=float(np.exp(s.loc[ai, "coef upper 95%"])), ai_p=float(s.loc[ai, "p"]))
    out["n_unknown_pT"] = int((~known).sum())
    out["unknown_pT_events"] = int(df.loc[~known, "recurred"].sum())
    return out


# ------------------------------------------------------------------ calibration
def km_risk(t, e, h=HORIZON):
    k = KaplanMeierFitter().fit(t, e)
    s = float(k.predict(h))
    lo = float(1 - k.confidence_interval_.iloc[:, 1][k.confidence_interval_.index <= h].iloc[-1])
    hi = float(1 - k.confidence_interval_.iloc[:, 0][k.confidence_interval_.index <= h].iloc[-1])
    return dict(n=int(len(t)), events=int(np.sum(e)), risk=1 - s, lo=lo, hi=hi)


def grade_risk(d, col):
    g = np.where(d[col] <= 2, "GG1-2", np.where(d[col] == 3, "GG3", "GG4-5"))
    return {lvl: km_risk(d.loc[g == lvl, "dfs_months"], d.loc[g == lvl, "recurred"]) for lvl in ("GG1-2", "GG3", "GG4-5")}


def model_calibration(d, cols, penalizer=0.0, reps=40):
    """Out-of-fold predicted 36-month risk vs Kaplan-Meier observed risk, in tertiles."""
    d = d[["dfs_months", "recurred"] + cols].dropna().reset_index(drop=True)
    n = len(d)
    pred = np.zeros((reps, n)); lp = np.zeros((reps, n))
    for rep in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(d, d["recurred"]):
            f = CoxPHFitter(penalizer=penalizer).fit(d.iloc[tr], "dfs_months", "recurred")
            s = f.predict_survival_function(d.iloc[te][cols], times=[HORIZON]).values.ravel()
            pred[rep, te] = 1 - s
            lp[rep, te] = f.predict_log_partial_hazard(d.iloc[te][cols]).values.ravel()
    p, l = pred.mean(0), lp.mean(0)
    slope = CoxPHFitter().fit(pd.DataFrame({"lp": l, "t": d["dfs_months"], "e": d["recurred"]}), "t", "e")
    r = slope.summary.loc["lp"]
    q = pd.qcut(pd.Series(p).rank(method="first"), 3, labels=["low", "mid", "high"])
    groups = {}
    for lvl in ("low", "mid", "high"):
        m = (q == lvl).values
        groups[lvl] = dict(mean_predicted=float(p[m].mean()), **km_risk(d.loc[m, "dfs_months"], d.loc[m, "recurred"]))
    obs = km_risk(d["dfs_months"], d["recurred"])
    return dict(n=int(n), slope=float(r["coef"]), slope_ci=[float(r["coef lower 95%"]), float(r["coef upper 95%"])],
                mean_predicted=float(p.mean()), observed=obs["risk"], oe_ratio=float(obs["risk"] / p.mean()), tertiles=groups)


# --------------------------------------------------------------- cohort table
def cohort(df):
    base = pd.read_csv(os.path.join(DATA, "clinical_baseline.csv"))
    d = df.merge(base, on="barcode", how="left")
    ev = d[d["recurred"] == 1]
    rk = KaplanMeierFitter().fit(d["dfs_months"], 1 - d["recurred"])  # reverse KM
    out = dict(
        n=int(len(d)), n_slides=int(d["n_slides"].sum()),
        single_slide=int((d["n_slides"] == 1).sum()),
        sites=int(d["barcode"].str.split("-").str[1].nunique()),
        age_median=float(d["age"].median()), age_iqr=[float(d["age"].quantile(.25)), float(d["age"].quantile(.75))],
        age_missing=int(d["age"].isna().sum()),
        race=d["RACE"].fillna("Not reported").value_counts().to_dict(),
        pT={k: int(v) for k, v in d["PATH_T_STAGE"].fillna("Unknown").value_counts().sort_index().items()},
        pN={k: int(v) for k, v in d["PATH_N_STAGE"].fillna("Not assessed / unknown").value_counts().items()},
        clin_gg={int(k): int(v) for k, v in d["clin_gg"].value_counts().sort_index().items()},
        clin_gg_missing=int(d["clin_gg"].isna().sum()),
        radiation={k: int(v) for k, v in d["RADIATION_THERAPY"].fillna("Unknown").value_counts().items()},
        events=int(d["recurred"].sum()),
        fu_median_all=float(d["dfs_months"].median()),
        fu_iqr_all=[float(d["dfs_months"].quantile(.25)), float(d["dfs_months"].quantile(.75))],
        fu_median_reverse_km=float(rk.median_survival_time_),
        fu_max=float(d["dfs_months"].max()),
        n_followed_past_60mo=int((d["dfs_months"] >= 60).sum()),
        time_to_event_median=float(ev["dfs_months"].median()),
        time_to_event_range=[float(ev["dfs_months"].min()), float(ev["dfs_months"].max())],
        events_within_36mo=int((ev["dfs_months"] <= 36).sum()),
        pfi_events=int((d["PFS_STATUS"].astype(str).str.startswith("1")).sum()),
        soft_n=int((d["soft_available"] == 1).sum()))
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {"B": B, "seed": SEED, "horizon_months": HORIZON, "minor_threshold": 0.05}
    dfi = load_rules(os.path.join(DATA, "features.csv"))
    both = dfi.dropna(subset=["clin_gg"]).copy()
    res["cohort"] = cohort(dfi)

    # ---- aggregation rules
    res["concordance"] = {r: concordance(both, c, rng) for r, c in (("area_ranked", "ai_gg_area"), ("isup_rp", "ai_gg_isup"))}
    res["thresholds"] = {r: thresholds(both, c) for r, c in (("area_ranked", "ai_gg_area"), ("isup_rp", "ai_gg_isup"))}
    res["minor_sweep"] = {}
    for m in (0.01, 0.02, 0.05, 0.10, 0.20):
        x = add_rules(both, m)
        c, a = x["clin_gg"].astype(int), x["ai_gg_isup"].astype(int)
        res["minor_sweep"][str(m)] = dict(kappa=float(cohen_kappa_score(c, a, weights="quadratic")),
                                          exact=float((c == a).mean()), within1=float((abs(c - a) <= 1).mean()),
                                          gg1_correct=int(((c == 1) & (a == 1)).sum()),
                                          gg2plus_called_gg1=int(((c >= 2) & (a == 1)).sum()),
                                          c_dfi=cstat(x["dfs_months"], x["recurred"], a))
    # decomposition of area-ranked discordance
    ar, isu, cl = both["ai_gg_area"], both["ai_gg_isup"], both["clin_gg"]
    res["decomposition"] = dict(
        discordant_area=int((ar != cl).sum()),
        resolved_by_isup=int(((ar != cl) & (isu == cl)).sum()),
        still_discordant=int(((ar != cl) & (isu != cl)).sum()),
        newly_discordant=int(((ar == cl) & (isu != cl)).sum()),
        changed_grade=int((ar != isu).sum()))
    # ---- high-grade share in over-called clinical GG1
    g1 = both[both["clin_gg"] == 1]
    oc = g1[g1["ai_gg_area"] >= 2]
    res["gg1_overcall"] = dict(
        n_gg1=int(len(g1)), n_overcalled=int(len(oc)), n_called_gg2=int((g1["ai_gg_area"] == 2).sum()),
        p4_share_quantiles={q: float(oc["pct_p4"].quantile(v)) for q, v in (("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("max", 1))},
        hg_share_quantiles={q: float(oc["pct_p4p5"].quantile(v)) for q, v in (("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("max", 1))},
        n_hg_le_1pct=int((oc["pct_p4p5"] <= .01).sum()), n_hg_le_5pct=int((oc["pct_p4p5"] <= .05).sum()),
        n_hg_le_10pct=int((oc["pct_p4p5"] <= .10).sum()), n_hg_gt_20pct=int((oc["pct_p4p5"] > .20).sum()),
        events_among_overcalled=int(oc["recurred"].sum()))
    g5 = both[both["clin_gg"] == 5]
    dr = g5[g5["ai_gg_area"] <= 3]
    res["gg5_undercall"] = dict(n_gg5=int(len(g5)), n_dropped=int(len(dr)),
                                dropped_median_p5=float(dr["pct_p5"].median()),
                                dropped_median_hg=float(dr["pct_p4p5"].median()),
                                dropped_min_hg=float(dr["pct_p4p5"].min()),
                                dropped_cribriform_pos=int((dr["cribriform_present"] == 1).sum()),
                                dropped_events=int(dr["recurred"].sum()),
                                kept_median_p5=float(g5[g5["ai_gg_area"] == 5]["pct_p5"].median()))

    # ---- endpoints
    res["endpoints"] = {}
    preds = ["clin_gg", "ai_gg_area", "ai_gg_isup", "pct_p4p5"]
    for ep, f in ENDPOINT_CSV.items():
        d = load_rules(os.path.join(DATA, f)).dropna(subset=["clin_gg"])
        d = d[d["dfs_months"] > 0]
        r = discrimination(d, rng, preds)
        r["logrank3"] = {p: logrank3(d, p) for p in ("clin_gg", "ai_gg_area", "ai_gg_isup")}
        med = d["pct_p4p5"].median()
        r["logrank_p4p5_median"] = float(logrank(d["dfs_months"], d["recurred"], (d["pct_p4p5"] > med).astype(int))[1])
        r["cox_unpenalized"] = {"clin_gg": unicox(d, "clin_gg"), "ai_gg_area": unicox(d, "ai_gg_area"),
                                "ai_gg_isup": unicox(d, "ai_gg_isup"), "pct_p4p5_perSD": unicox(d, "pct_p4p5", std=True),
                                "pt_stage": unicox(d, "pt_stage_num")}
        r["cox_ridge0.1"] = {"clin_gg": unicox(d, "clin_gg", penalizer=.1), "ai_gg_area": unicox(d, "ai_gg_area", penalizer=.1),
                             "pct_p4p5_perSD": unicox(d, "pct_p4p5", std=True, penalizer=.1),
                             "pt_stage": unicox(d, "pt_stage_num", penalizer=.1)}
        r["joint"] = {"clin+ai_area": joint(d, "clin_gg", "ai_gg_area"), "clin+ai_isup": joint(d, "clin_gg", "ai_gg_isup"),
                      "clin+pct_p4p5": joint(d, "clin_gg", "pct_p4p5"), "ai_area+pT": joint(d, "ai_gg_area", "pt_stage_num"),
                      "clin+pT": joint(d, "clin_gg", "pt_stage_num")}
        r["cribriform_perSD"] = unicox(d, "cribriform_area_mm2", std=True)
        res["endpoints"][ep] = r

    # ---- optimism + imputation
    res["optimism"] = {"ai_gg_area+pT": optimism(dfi, ["ai_gg_area", "pt_stage_num"], rng),
                       "ai_gg_isup+pT": optimism(dfi, ["ai_gg_isup", "pt_stage_num"], rng),
                       "pct_p4p5+pT": optimism(dfi, ["pct_p4p5", "pt_stage_num"], rng),
                       "clin_gg+pT": optimism(both, ["clin_gg", "pt_stage_num"], rng)}
    res["pt_sensitivity"] = pt_sensitivity(dfi, "ai_gg_area")

    # ---- calibration
    res["grade_risk_36mo"] = {"clinical": grade_risk(both, "clin_gg"), "ai_area_ranked": grade_risk(both, "ai_gg_area"),
                              "ai_isup_rp": grade_risk(both, "ai_gg_isup"), "overall": km_risk(dfi["dfs_months"], dfi["recurred"])}
    res["model_calibration"] = {"pct_p4p5": model_calibration(dfi, ["pct_p4p5"]),
                                "ai_gg_area+pT": model_calibration(dfi, ["ai_gg_area", "pt_stage_num"])}

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
