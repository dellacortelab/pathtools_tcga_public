#!/usr/bin/env python3
"""Download the clinical variables used for Table 1 and the adverse-pathology analysis
from cBioPortal -> data/clinical_baseline.csv."""
import os

import pandas as pd
import requests

DATA = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data")
CB, STUDY = "https://www.cbioportal.org/api", "prad_tcga_pan_can_atlas_2018"
ATTRS = ["RACE", "ETHNICITY", "AGE", "SEX", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE",
         "AJCC_PATHOLOGIC_TUMOR_STAGE", "OS_STATUS", "OS_MONTHS", "DSS_STATUS", "DFS_STATUS", "DFS_MONTHS",
         "PFS_STATUS", "PFS_MONTHS", "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
         "PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT", "RADIATION_THERAPY", "PERSON_NEOPLASM_CANCER_STATUS",
         "HISTORY_NEOADJUVANT_TRTYN"]


def attr(a):
    r = requests.get(f"{CB}/studies/{STUDY}/clinical-data",
                     params={"clinicalDataType": "PATIENT", "attributeId": a, "projection": "SUMMARY",
                             "pageSize": "100000"}, timeout=60)
    r.raise_for_status()
    return {d["patientId"]: d["value"] for d in r.json()}


df = pd.DataFrame({"barcode": sorted(pd.read_csv(os.path.join(DATA, "features.csv"))["barcode"])})
for a in ATTRS:
    df[a] = df["barcode"].map(attr(a))
df.to_csv(os.path.join(DATA, "clinical_baseline.csv"), index=False)
print(df.shape)
