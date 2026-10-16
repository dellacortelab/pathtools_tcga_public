#!/usr/bin/env python3
"""All data figures (main Figures 2-5, Supplementary Figures S1-S6), written to
figures/ as vector PDF, 600 dpi TIFF (LZW) and 600 dpi PNG. Every plotted number
comes from results/*.json or is computed from the feature tables in data/.

Run:  python make_figures.py      (after analysis.py and the two added_value scripts)
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from rules import load_rules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RESULTS, FIGURES, kaplan_meier, logrank  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
OUT = FIGURES
STATS = json.load(open(os.path.join(RESULTS, "analysis.json")))

INK, MUTED, GRID = "#1a1a1a", "#5f5f5f", "#d9d9d9"
C_CLIN, C_AI, C_AI2, C_PCT = "#D55E00", "#0072B2", "#56B4E9", "#009E73"
C_LO, C_MID, C_HI = "#009E73", "#E69F00", "#CC3311"
DOUBLE = 7.0  # inches (178 mm, double column)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, name)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".tiff", dpi=600, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", base + ".{pdf,png,tiff}")


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}" if letter else title, loc="left", fontweight="bold", pad=5)


# ------------------------------------------------------------------ Figure 2
def fig2():
    conc = STATS["concordance"]
    fig = plt.figure(figsize=(DOUBLE, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.38)
    for k, (rule, letter, title) in enumerate((("area_ranked", "A", "Area-ranked rule"),
                                               ("isup_rp", "B", "ISUP rule (5% threshold)"))):
        ax = fig.add_subplot(gs[k])
        M = np.array(conc[rule]["confusion"], float)
        Mn = M / M.sum(1, keepdims=True)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(5):
            for j in range(5):
                if M[i, j]:
                    ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=7,
                            color="white" if Mn[i, j] > 0.55 else INK)
            ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=INK, lw=0.7))
        ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6))
        ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
        ax.set_xlabel("AI grade group")
        if k == 0:
            ax.set_ylabel("Clinical grade group")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        panel(ax, letter, title)
        c = conc[rule]
        ax.text(0.5, -0.30, f"κ = {c['kappa']:.2f} (95% CI {c['kappa_ci'][0]:.2f}–{c['kappa_ci'][1]:.2f})\n"
                            f"exact {c['exact']:.0%}, within one group {c['within1']:.0%}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color=MUTED, linespacing=1.4)
    ax = fig.add_subplot(gs[2])
    xs = np.arange(1, 6)
    ax.plot([1, 5], [1, 5], ls=(0, (4, 3)), color=MUTED, lw=0.8)
    ax.text(1.55, 1.2, "identity", fontsize=6.5, color=MUTED, rotation=45, ha="center", va="center")
    for rule, col, mk, lab, off in (("area_ranked", C_AI, "o", "Area-ranked", -0.07), ("isup_rp", C_AI2, "s", "ISUP rule", 0.07)):
        m, sd = np.array(conc[rule]["mean_ai_by_clin"]), np.array(conc[rule]["sd_ai_by_clin"])
        ax.errorbar(xs + off, m, yerr=sd, color=col, marker=mk, ms=4.2, mec="white", mew=0.6, lw=1.3,
                    capsize=1.8, elinewidth=0.8, label=lab)
    ax.set_xticks(xs); ax.set_yticks(xs)
    ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5); ax.set_aspect("equal")
    ax.set_xlabel("Clinical grade group"); ax.set_ylabel("AI grade group (mean ± SD)")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6, borderaxespad=0.2)
    panel(ax, "C", "Mean AI grade")
    save(fig, "Figure2_concordance")


# ------------------------------------------------------------------ KM figures
def gg_bin(s):
    return np.where(s <= 2, "GG 1–2", np.where(s == 3, "GG 3", "GG 4–5"))


def km_axes(fig, gs_cell, d, group, order, colors, styles, title, letter, ylabel, xmax=60, ymin=0.5):
    sub = gs_cell.subgridspec(2, 1, height_ratios=[4.2, 1.15], hspace=0.17)
    ax, tb = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    t_all, e_all = d["dfs_months"].values, d["recurred"].values
    p = logrank(t_all, e_all, group)[1]
    ticks = np.arange(0, xmax + 1, 12)
    for k, lvl in enumerate(order):
        m = group == lvl
        t, s = kaplan_meier(t_all[m], e_all[m])
        keep = t <= xmax
        ax.step(np.r_[t[keep], xmax], np.r_[s[keep], s[keep][-1]], where="post", color=colors[lvl], lw=1.4,
                ls=styles[lvl], label=f"{lvl} (n = {int(m.sum())}, {int(e_all[m].sum())} events)")
        cens = (e_all[m] == 0) & (t_all[m] <= xmax)
        ct = t_all[m][cens]
        ax.plot(ct, np.interp(ct, t, s, left=1) if False else [s[np.searchsorted(t, x, side="right") - 1] for x in ct],
                "|", color=colors[lvl], ms=3, mew=0.5, alpha=0.7)
        for j, x in enumerate(ticks):
            tb.text(x, len(order) - 1 - k, str(int((t_all[m] >= x).sum())), ha="center", va="center", fontsize=6.3)
        tb.plot([-0.14], [len(order) - 1 - k], marker="s", ms=3.5, color=colors[lvl], clip_on=False,
                transform=tb.get_yaxis_transform())
    ax.set_xlim(0, xmax); ax.set_ylim(ymin, 1.01); ax.set_xticks(ticks); ax.set_xticklabels([])
    ax.yaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="lower left", fontsize=6.4, handlelength=1.8, borderaxespad=0.5, labelspacing=0.25)
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.04, 0.07 + 0.085 * len(order) + 0.03, f"Log-rank {ptxt}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8)
    panel(ax, letter, title)
    tb.set_xlim(0, xmax); tb.set_ylim(-0.6, len(order) - 0.4); tb.set_xticks(ticks); tb.set_yticks([])
    for s_ in ("left", "top", "right"):
        tb.spines[s_].set_visible(False)
    tb.set_xlabel("Months since prostatectomy")
    tb.text(0.0, 1.0, "Number at risk", transform=tb.transAxes, ha="left", va="bottom", fontsize=6.0, color=MUTED)
    return ax


def km_figure(csv, name, ylabel, variant="main"):
    df = load_rules(os.path.join(DATA, csv))
    df = df[df["dfs_months"] > 0]
    d = df.dropna(subset=["clin_gg"])
    three = {"GG 1–2": C_LO, "GG 3": C_MID, "GG 4–5": C_HI}
    st3 = {"GG 1–2": "-", "GG 3": (0, (5, 1.5)), "GG 4–5": (0, (1.2, 1.2))}
    order3 = list(three)
    fig = plt.figure(figsize=(DOUBLE, 3.15))
    gs = fig.add_gridspec(1, 3, wspace=0.28)
    km_axes(fig, gs[0], d, gg_bin(d["clin_gg"].values), order3, three, st3, "Clinical grade group", "A", ylabel)
    if variant == "main":
        km_axes(fig, gs[1], d, gg_bin(d["ai_gg_area"].values), order3, three, st3, "AI grade group", "B", None)
        med = df["pct_p4p5"].median()
        g = np.where(df["pct_p4p5"].values > med, "Above median", "At or below median")
        two = {"At or below median": C_LO, "Above median": C_HI}
        km_axes(fig, gs[2], df, g, list(two), two, {"At or below median": "-", "Above median": (0, (1.2, 1.2))},
                "AI % Gleason 4+5", "C", None)
    else:  # both aggregation rules side by side
        km_axes(fig, gs[1], d, gg_bin(d["ai_gg_area"].values), order3, three, st3, "AI grade, area-ranked", "B", None)
        km_axes(fig, gs[2], d, gg_bin(d["ai_gg_isup"].values), order3, three, st3, "AI grade, ISUP rule", "C", None)
    save(fig, name)


# ------------------------------------------------------------------ Figure 4
def fmt(v):
    """Signed two-decimal number without a negative zero."""
    v = round(v, 2)
    return "0.00" if v == 0 else f"{v:+.2f}"


def fig4():
    eps = STATS["endpoints"]
    rows = [("clin_gg", "Clinical grade group", C_CLIN, "D"), ("ai_gg_area", "AI grade group\n(area-ranked)", C_AI, "o"),
            ("ai_gg_isup", "AI grade group\n(ISUP rule)", C_AI2, "s"), ("pct_p4p5", "AI % Gleason 4+5", C_PCT, "^")]
    titles = {"DFI": "Disease-free interval", "PFI": "Progression-free interval", "BCR": "Biochemical (PSA) recurrence"}
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    ys = np.arange(len(rows))[::-1]
    for ax, letter, ep in zip(axes, "ABC", ("DFI", "PFI", "BCR")):
        e = eps[ep]
        ref = e["clin_gg"]
        ax.axvspan(ref["ci"][0], ref["ci"][1], color=C_CLIN, alpha=0.08, lw=0)
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        for y, (key, lab, col, mk) in zip(ys, rows):
            r = e[key]
            ax.plot(r["ci"], [y, y], color=col, lw=1.6, solid_capstyle="round")
            ax.plot(r["c"], y, mk, color=col, ms=5.2, mec="white", mew=0.7)
            txt = f"{r['c']:.2f} ({r['ci'][0]:.2f}–{r['ci'][1]:.2f})"
            ax.text(r["c"], y + 0.27, txt, ha="center", va="bottom", fontsize=6.3)
            if "delta_ci" in r:
                ax.text(1.0, y - 0.3, f"Δ {fmt(r['delta_clin_minus'])} ({fmt(r['delta_ci'][0])} to {fmt(r['delta_ci'][1])})",
                        ha="right", va="top", fontsize=6.0, color=MUTED, transform=ax.get_yaxis_transform())
        ax.set_xlim(0.45, 0.92); ax.set_ylim(-0.75, len(rows) - 0.3)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_xlabel("Harrell c-index (95% CI)")
        ax.xaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
        panel(ax, letter, f"{titles[ep]}")
        ax.text(0.0, 1.005, f"n = {e['n']}, {e['events']} events", transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom")
        ax.title.set_position((0, 1.06)); ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=13)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([r[1] for r in rows])
    for ax in axes[1:]:
        ax.tick_params(left=False)
    save(fig, "Figure4_discrimination")


# ------------------------------------------------------------------ Figure 5
def fig5():
    gr, mc = STATS["grade_risk_36mo"], STATS["model_calibration"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3})
    strata = ["GG1-2", "GG3", "GG4-5"]
    series = [("clinical", "Clinical grade", C_CLIN, "D", -0.2), ("ai_area_ranked", "AI grade, area-ranked", C_AI, "o", 0.0),
              ("ai_isup_rp", "AI grade, ISUP rule", C_AI2, "s", 0.2)]
    for key, lab, col, mk, off in series:
        for i, s in enumerate(strata):
            r = gr[key][s]
            a.plot([i + off] * 2, [100 * r["lo"], 100 * r["hi"]], color=col, lw=1.3, solid_capstyle="round")
            a.plot(i + off, 100 * r["risk"], mk, color=col, ms=5, mec="white", mew=0.7, label=lab if i == 0 else None)
            a.text(i + off, 100 * r["hi"] + 0.8, f"{r['events']}/{r['n']}", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    a.axhline(100 * gr["overall"]["risk"], color=MUTED, lw=0.6, ls=(0, (4, 3)))
    a.text(2.45, 100 * gr["overall"]["risk"] + 0.5, "cohort", fontsize=6.2, color=MUTED, ha="right", va="bottom")
    a.set_xticks(range(3)); a.set_xticklabels(["GG 1–2", "GG 3", "GG 4–5"])
    a.set_xlim(-0.5, 2.5); a.set_ylim(0, 36)
    a.set_ylabel("Observed 36-month event risk, %\n(Kaplan–Meier, 95% CI)"); a.set_xlabel("Assigned grade stratum")
    a.yaxis.grid(True, color=GRID, lw=0.4); a.set_axisbelow(True)
    a.legend(frameon=False, loc="upper left", borderaxespad=0.2)
    a.text(0.985, 0.03, "events/patients above each interval", transform=a.transAxes, ha="right", fontsize=5.8, color=MUTED)
    panel(a, "A", "Risk carried by each assigned grade")

    lim = 30
    b.plot([0, lim], [0, lim], color=MUTED, lw=0.7, ls=(0, (4, 3)))
    for key, lab, col, mk in (("ai_gg_area+pT", "AI grade + pT stage", C_AI, "o"), ("pct_p4p5", "AI % Gleason 4+5", C_PCT, "^")):
        t = mc[key]["tertiles"]
        x = [100 * t[k]["mean_predicted"] for k in ("low", "mid", "high")]
        y = [100 * t[k]["risk"] for k in ("low", "mid", "high")]
        for xi, k in zip(x, ("low", "mid", "high")):
            b.plot([xi, xi], [100 * t[k]["lo"], 100 * t[k]["hi"]], color=col, lw=1.0, alpha=0.75)
        b.plot(x, y, marker=mk, color=col, ms=5, mec="white", mew=0.7, lw=1.0,
               label=f"{lab}\nslope {mc[key]['slope']:.2f} ({mc[key]['slope_ci'][0]:.2f}–{mc[key]['slope_ci'][1]:.2f})")
    b.set_xlim(0, lim); b.set_ylim(0, lim); b.set_aspect("equal")
    b.set_xlabel("Cross-validated predicted 36-month risk, %"); b.set_ylabel("Observed 36-month risk, %")
    b.xaxis.grid(True, color=GRID, lw=0.4); b.yaxis.grid(True, color=GRID, lw=0.4); b.set_axisbelow(True)
    b.legend(frameon=False, loc="upper left", borderaxespad=0.2, labelspacing=0.6)
    panel(b, "B", "Calibration of fitted models (tertiles)")
    save(fig, "Figure5_calibration")


# ------------------------------------------------------- Supplementary figures
def figS1_flow():
    c = STATS["cohort"]
    bcr = STATS["endpoints"]["BCR"]
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f6f8", bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=INK, lw=0.7))
        ax.text(x, y, text, ha="center", va="center", fontsize=7, fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, shrinkA=0, shrinkB=0))

    X = 3.6
    box(X, 9.3, 5.4, 0.9, "TCGA-PRAD cases in the NCI Genomic Data Commons\nn = 500 patients (1,172 open-access slide images)")
    box(X, 7.5, 5.4, 0.9, "Patients with ≥ 1 diagnostic FFPE (DX) whole-slide image\nn = 403 patients (449 slides)")
    box(X, 5.7, 5.4, 0.9, f"Disease-free-interval outcome recorded (TCGA-CDR)\nn = {c['n']} patients ({c['n_slides']} slides; {c['sites']} tissue source sites)")
    box(X, 3.9, 5.4, 0.9, f"Automated grading completed, no slide excluded\nn = {c['n']} patients ({c['n_slides']} slides; {c['single_slide']} patients with one slide)", fc="#e8f1f8")
    for y0, y1 in ((8.85, 7.95), (7.05, 6.15), (5.25, 4.35)):
        arrow(X, y0, X, y1)
    box(8.3, 8.4, 3.2, 0.75, "Excluded: frozen-section slides only\nn = 97 patients", fc="white")
    box(8.3, 6.6, 3.2, 0.75, "Excluded: no disease-free-interval\nrecord, n = 129 patients (151 slides)", fc="white")
    arrow(X, 8.4, 6.7, 8.4); arrow(X, 6.6, 6.7, 6.6)
    ys = 1.55
    box(1.25, ys, 2.3, 1.5, f"Prognostic analysis\nDFI (primary)\nn = {c['n']}, {c['events']} events", fc="#e8f1f8", bold=False)
    box(3.75, ys, 2.3, 1.5, f"Concordance with\nclinical grade\nn = {STATS['concordance']['area_ranked']['n']}\n({c['clin_gg_missing']} lack a clinical grade)", fc="#e8f1f8")
    box(6.25, ys, 2.3, 1.5, f"Sensitivity: PFI\nn = {c['n']}, {c['pfi_events']} events", fc="#e8f1f8")
    box(8.75, ys, 2.3, 1.5, f"Sensitivity: PSA\nrecurrence\nn = {bcr['n'] + 1}, {bcr['events']} events\n(40 without usable record)", fc="#e8f1f8")
    ax.plot([1.25, 8.75], [2.75, 2.75], color=INK, lw=0.7)
    ax.plot([X, X], [3.45, 2.75], color=INK, lw=0.7)
    for x in (1.25, 3.75, 6.25, 8.75):
        arrow(x, 2.75, x, 2.32)
    save(fig, "FigureS1_patient_flow")


def figS_sweep():
    sw = STATS["minor_sweep"]
    ths = sorted(sw, key=float)
    x = np.arange(len(ths))
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), gridspec_kw={"wspace": 0.32})
    a.plot(x, [sw[t]["kappa"] for t in ths], "o-", color=C_AI2, ms=4.5, mec="white", lw=1.2, label="Quadratic-weighted κ")
    a.plot(x, [sw[t]["exact"] for t in ths], "s-", color=C_PCT, ms=4.5, mec="white", lw=1.2, label="Exact agreement")
    a.plot(x, [sw[t]["c_dfi"] for t in ths], "^-", color=MUTED, ms=4.5, mec="white", lw=1.2, label="c-index, DFI")
    a.axhline(STATS["concordance"]["area_ranked"]["kappa"], color=C_AI, lw=0.7, ls=(0, (4, 3)))
    a.text(len(ths) - 1, STATS["concordance"]["area_ranked"]["kappa"] - 0.012, "κ, area-ranked rule", ha="right", va="top", fontsize=6.2, color=MUTED)
    a.set_ylim(0.0, 0.8); a.set_ylabel("Agreement / discrimination")
    panel(a, "A", "Agreement across thresholds")
    b.plot(x, [sw[t]["gg1_correct"] for t in ths], "o-", color=C_LO, ms=4.5, mec="white", lw=1.2, label="Clinical GG1 graded GG1 (of 30)")
    b.plot(x, [sw[t]["gg2plus_called_gg1"] for t in ths], "s-", color=C_HI, ms=4.5, mec="white", lw=1.2, label="Clinical GG ≥ 2 graded GG1 (of 242)")
    b.set_ylabel("Patients"); b.set_ylim(0, 60)
    panel(b, "B", "Errors at the GG1 boundary")
    for ax in (a, b):
        ax.set_xticks(x); ax.set_xticklabels([f"{float(t):.0%}" for t in ths])
        ax.set_xlabel("Minor-pattern threshold (share of tumour)")
        ax.yaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper left" if ax is b else "lower left", borderaxespad=0.2)
    save(fig, "FigureS5_threshold_sweep")


# ------------------------------------------------------------------ Supplementary Figure S6
def fig6():
    """Every AI-derived candidate, adjusted for the clinical grade group, on three outcomes."""
    av = json.load(open(os.path.join(RESULTS, "added_value.json")))
    ap = json.load(open(os.path.join(RESULTS, "added_value_pathology.json")))
    n1_key = next(k for k in ap if k.startswith("N1"))
    panels = [("A", "Disease-free interval", f"n = {av['DFI']['n']}, {av['DFI']['events']} events", "Hazard ratio",
               {k: (v["hr"], v["lr_linear_bh"]) for k, v in av["DFI"]["tests"].items()}),
              ("B", "Extraprostatic disease", f"n = {ap['pT3 or higher']['n']}, {ap['pT3 or higher']['events']} with pT3–4", "Odds ratio",
               {k: (v["odds_ratio"], v["lr_p_bh"]) for k, v in ap["pT3 or higher"]["tests"].items()}),
              ("C", "Nodal metastasis", f"n = {ap[n1_key]['n']}, {ap[n1_key]['events']} with N1", "Odds ratio",
               {k: (v["odds_ratio"], v["lr_p_bh"]) for k, v in ap[n1_key]["tests"].items()})]
    names = list(av["DFI"]["tests"])
    labels = [n.replace("AI ", "").replace("tumour", "tumor").replace("mm2", "mm²") for n in names]
    labels = [l[0].upper() + l[1:] for l in labels]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 3.9), sharey=True)
    fig.subplots_adjust(wspace=0.1)
    ys = np.arange(len(names))[::-1]
    lo_lim, hi_lim = 0.3, 10
    for ax, (letter, title, sub, xl, data) in zip(axes, panels):
        ax.axvline(1, color=MUTED, lw=0.7, ls=(0, (4, 3)))
        for y, n in zip(ys, names):
            (est, lo, hi), q = data[n]
            sig = q < 0.05
            col = C_AI if sig else MUTED
            ax.plot([max(lo, lo_lim), min(hi, hi_lim)], [y, y], color=col, lw=1.5 if sig else 0.9, solid_capstyle="round")
            ax.plot(min(max(est, lo_lim), hi_lim), y, "o", ms=4.8, color=col, mfc=col if sig else "white", mec=col, mew=1.0)
        ax.set_xscale("log"); ax.set_xlim(lo_lim, hi_lim)
        ax.set_xticks([0.5, 1, 2, 5]); ax.set_xticklabels(["0.5", "1", "2", "5"]); ax.minorticks_off()
        ax.set_xlabel(f"{xl} (95% CI)")
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.xaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=13)
        ax.text(0.0, 1.005, sub, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom")
    axes[0].set_yticks(ys); axes[0].set_yticklabels(labels)
    for ax in axes[1:]:
        ax.tick_params(left=False)
    fig.legend(handles=[Line2D([0], [0], marker="o", color=C_AI, mfc=C_AI, lw=1.4, ms=4.8, label="FDR-adjusted p < 0.05"),
                        Line2D([0], [0], marker="o", color=MUTED, mfc="white", lw=1.0, ms=4.8, label="Not significant")],
               frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.06), handlelength=1.8)
    save(fig, "FigureS6_added_value")


if __name__ == "__main__":
    fig2()
    km_figure("features.csv", "Figure3_kaplan_meier_DFI", "Disease-free probability")
    fig4()
    fig5()
    figS1_flow()
    km_figure("features_pfs.csv", "FigureS2_kaplan_meier_PFI", "Progression-free probability")
    km_figure("features_psa.csv", "FigureS3_kaplan_meier_BCR", "PSA-recurrence-free probability")
    km_figure("features.csv", "FigureS4_kaplan_meier_DFI_aggregation_rules", "Disease-free probability", variant="rules")
    figS_sweep()
    fig6()
