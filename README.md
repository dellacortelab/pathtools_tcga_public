# Automated Gleason grading on TCGA prostatectomy slides: data and analysis code

Feature tables and analysis code for *An Automated, Pathologist-free Gleason Grade Carries
Standalone Prognostic Signal on Out-of-distribution Prostatectomy Slides: A Developer-led
Proof-of-concept Evaluation in TCGA*. Every statistic and every data figure in the article
and its Supporting Information can be recomputed from this repository.

The grading system evaluated, PathTools Prostate v11.0, is proprietary and is not included.
This repository holds its outputs as tabulated features, not the model, its slide-level
annotations, or any whole-slide images.

## Layout

```
data/       input tables (below)
scripts/    analysis and figure code
results/    written by the scripts (JSON)
figures/    written by make_figures.py (PDF, 600 dpi PNG and TIFF)
```

| File | Content |
|---|---|
| `data/features_slide.csv` | One row per diagnostic slide (298): automated tissue and tumor area, pattern 3/4/5 areas and shares, gland counts, cribriform and perineural-invasion detections, tumor foci |
| `data/features.csv` | One row per patient (274), slides pooled, joined to the disease-free-interval outcome, clinical Gleason score, pathologic stage and age |
| `data/features_pfs.csv` | As `features.csv`, with the progression-free interval as outcome |
| `data/features_psa.csv` | As `features.csv`, with biochemical (PSA) recurrence as outcome (234 patients) |
| `data/clinical_baseline.csv` | Public clinical variables from cBioPortal (cohort table, nodal stage) |
| `scripts/common.py` | Paths, feature loading, clinical grade group, Kaplan-Meier and log-rank |
| `scripts/rules.py` | The two grade-group aggregation rules: area-ranked, and ISUP with a minor-pattern threshold |
| `scripts/analysis.py` | Concordance, discrimination, Cox models, calibration, optimism correction and sensitivity analyses → `results/analysis.json` |
| `scripts/added_value.py` | Exploratory analysis of information beyond the clinical grade for recurrence → `results/added_value.json` |
| `scripts/added_value_pathology.py` | The same for extraprostatic disease and nodal metastasis → `results/added_value_pathology.json` |
| `scripts/make_figures.py` | Figures 2–5 and Supplementary Figures S1–S6 → `figures/` |
| `scripts/pull_clinical.py` | Re-downloads `data/clinical_baseline.csv` from the cBioPortal API |

## Data sources

Whole-slide images: NCI Genomic Data Commons, project TCGA-PRAD, diagnostic (`-DX`) slides.
Outcomes and clinical variables: TCGA Pan-Cancer Clinical Data Resource via cBioPortal
(`prad_tcga_pan_can_atlas_2018`; PSA recurrence from `prad_tcga`). Patients are identified by
the public TCGA case barcode only.

## Reproducing the analysis

Python 3.9 or later with numpy, pandas, scipy, scikit-learn, lifelines and matplotlib
(`requests` only for `pull_clinical.py`).

```bash
cd scripts
python analysis.py                # about 10 minutes (2,000 bootstrap replicates)
python added_value.py             # about 30 minutes
python added_value_pathology.py
python make_figures.py
```

All resampling uses fixed seeds, so the outputs are deterministic. Set `B=200` in the
environment for a quick run of `analysis.py` with fewer bootstrap replicates.

## License

Code is released under the MIT License (see `LICENSE`). The clinical data are TCGA open-access
data, subject to the NCI Genomic Data Commons data-use policies.
