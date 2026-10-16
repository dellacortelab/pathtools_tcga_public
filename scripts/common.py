"""Shared helpers: paths, feature-table loading, clinical grade group, and a
numpy/scipy Kaplan-Meier estimator and k-group log-rank test."""
import os
import re
import numpy as np
import pandas as pd

# repository layout: data/ (inputs), results/ and figures/ (outputs), scripts/ (this file)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "figures")
FEATURES = os.environ.get("FEATURES_CSV", os.path.join(DATA, "features.csv"))

# Colour palette (colour-blind safe, consistent across all figures)
C_AI = "#0072B2"       # AI / model
C_CLIN = "#D55E00"     # clinical / pathologist
C_HI = "#CC3311"       # high-risk stratum
C_LO = "#009E73"       # low-risk stratum
GREY = "#888888"


def clinical_grade_group(clin_gleason: str) -> float:
    """ISUP grade group from the TCGA clinical Gleason string, e.g.
    'Pattern 4+Pattern 5' -> 5. Returns nan if unparseable."""
    m = re.findall(r"[Pp]attern\s*(\d)", str(clin_gleason))
    if len(m) < 2:
        return np.nan
    a, b = int(m[0]), int(m[1])
    s = a + b
    if s <= 6:
        return 1.0
    if (a, b) == (3, 4):
        return 2.0
    if (a, b) == (4, 3):
        return 3.0
    if s == 8:
        return 4.0
    return 5.0


def load(features_csv: str = FEATURES) -> pd.DataFrame:
    """Load the per-patient feature table with derived clinical GG and clean
    survival columns. AI grade group is `grade_group_area` (area-ranked
    primary+secondary)."""
    df = pd.read_csv(features_csv)
    df["clin_gg"] = df["clin_gleason"].apply(clinical_grade_group)
    df = df.dropna(subset=["dfs_months"]).copy()
    df["dfs_months"] = df["dfs_months"].astype(float)
    df["recurred"] = df["recurred"].astype(int)
    df["ai_gg"] = df["grade_group_area"].astype(float)
    return df


def kaplan_meier(time, event):
    """Return (t, surv) step coordinates for the KM estimator."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    order = np.argsort(time)
    time, event = time[order], event[order]
    uniq = np.unique(time)
    n = len(time)
    surv, s, t_out, s_out = 1.0, 1.0, [0.0], [1.0]
    for ut in uniq:
        at_risk = np.sum(time >= ut)
        d = np.sum((time == ut) & (event == 1))
        if at_risk > 0 and d > 0:
            s *= (1 - d / at_risk)
        t_out.append(ut)
        s_out.append(s)
    return np.array(t_out), np.array(s_out)


def logrank(time, event, group):
    """k-group log-rank test. Returns (chi2, p, dof) via scipy chi2 sf."""
    from scipy.stats import chi2 as _chi2
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    group = np.asarray(group)
    groups = np.unique(group)
    g = len(groups)
    # observed events per group
    O = np.array([event[(group == gr)].sum() for gr in groups], float)
    E = np.zeros(g)
    # build variance-covariance across the distinct event times
    death_times = np.unique(time[event == 1])
    V = np.zeros((g, g))
    for dt in death_times:
        at_risk_total = np.sum(time >= dt)
        d_total = np.sum((time == dt) & (event == 1))
        if at_risk_total <= 1:
            continue
        n_j = np.array([np.sum((time >= dt) & (group == gr)) for gr in groups], float)
        E += d_total * n_j / at_risk_total
        f = d_total * (at_risk_total - d_total) / (at_risk_total ** 2 * (at_risk_total - 1))
        for i in range(g):
            for j in range(g):
                delta = 1.0 if i == j else 0.0
                V[i, j] += f * (delta * n_j[i] * at_risk_total - n_j[i] * n_j[j])
    diff = (O - E)[:-1]
    Vred = V[:-1, :-1]
    try:
        chi2_stat = float(diff @ np.linalg.solve(Vred, diff))
    except np.linalg.LinAlgError:
        chi2_stat = float(diff @ np.linalg.pinv(Vred) @ diff)
    dof = g - 1
    p = float(_chi2.sf(chi2_stat, dof))
    return chi2_stat, p, dof

