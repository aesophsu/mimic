# Table 2. Internal and External Validation Performance Matrix

| Endpoints & Model | Internal Validation (MIMIC-IV) AUROC (95% CI) | Internal Validation Brier | External Validation (eICU-CRD) AUROC (95% CI) | External Validation Brier |
|---|---:|---:|---:|---:|
| **Primary Endpoint (POF)** |  |  |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;Full-feature XGBoost (n=12) | 0.845 (0.781-0.895) | 0.157 | 0.847 (0.824-0.870) | 0.165 |
| &nbsp;&nbsp;&nbsp;&nbsp;Parsimonious XGBoost (n=3) | 0.848 (0.795-0.901) | 0.156 | **0.839 (0.815-0.862)** | **0.175** |
| &nbsp;&nbsp;&nbsp;&nbsp;Random Forest (n=12) | 0.848 (0.788-0.899) | 0.156 | 0.859 (0.837-0.880) | 0.164 |
| &nbsp;&nbsp;&nbsp;&nbsp;SVM (n=12) | 0.853 (0.796-0.903) | 0.151 | 0.848 (0.826-0.871) | 0.161 |
| &nbsp;&nbsp;&nbsp;&nbsp;Logistic Regression (n=12) | 0.846 (0.787-0.898) | 0.153 | 0.736 (0.706-0.767) | 0.210 |
| &nbsp;&nbsp;&nbsp;&nbsp;Decision Tree (n=12) | 0.839 (0.777-0.890) | 0.161 | 0.824 (0.799-0.848) | 0.176 |
| **Secondary Endpoint (28-day Mortality)** |  |  |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;Full-feature XGBoost (n=12) | 0.857 (0.782-0.917) | 0.100 | 0.841 (0.804-0.876) | 0.064 |
| &nbsp;&nbsp;&nbsp;&nbsp;Parsimonious XGBoost (n=6) | 0.809 (0.736-0.881) | 0.110 | **0.847 (0.812-0.881)** | **0.068** |
| &nbsp;&nbsp;&nbsp;&nbsp;Logistic Regression (n=12) | 0.876 (0.812-0.929) | 0.098 | 0.810 (0.767-0.852) | 0.067 |
| **Composite Endpoint** |  |  |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;Full-feature XGBoost (n=12) | 0.866 (0.816-0.913) | 0.150 | 0.857 (0.834-0.878) | 0.160 |
| &nbsp;&nbsp;&nbsp;&nbsp;Parsimonious XGBoost (n=4) | 0.866 (0.822-0.913) | 0.147 | **0.852 (0.828-0.872)** | **0.166** |

Notes:
- For the primary endpoint (POF), all developed algorithms are presented for a comprehensive comparison. For secondary and composite endpoints, only the top-performing models and their corresponding parsimonious versions are shown to maintain conciseness and emphasize clinical utility.
- AUROC values are shown with 95% CI.
- Brier score reflects probabilistic calibration (lower is better).
- Abbreviations: AUROC, area under the receiver operating characteristic curve; LR, logistic regression.
- Parsimonious feature sets:
  - POF (n=3): `P/F ratio (minimum); Creatinine (maximum); pH (minimum)`
  - 28-day mortality (n=6): `Age at admission; PTT (minimum); Blood urea nitrogen (maximum); Lactate (maximum); Hemoglobin (minimum); Albumin (maximum)`
  - Composite endpoint (n=4): `P/F ratio (minimum); Creatinine (maximum); pH (minimum); Blood urea nitrogen (minimum)`