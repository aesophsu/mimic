# Fig5 Notes

- Case A: True Positive with high predicted POF risk.
- Case B: True Negative with low predicted POF risk.
- Red bars increase risk; blue bars decrease risk.
- Units were harmonized to Table1 conventions where applicable (e.g., Creatinine in umol/L, Urea in mmol/L, P/F ratio shown without unit).
- Missing values among displayed top features: Case A = 0, Case B = 0. If non-zero, missingness was handled by model-imputation pipeline (median/MICE as configured upstream).
- Case A interpretation: Primary risk driver: Decreased P/F ratio. Risk mitigated by other physiological factors.
- Case B interpretation: Primary risk driver: Combined adverse contribution from other physiological factors. Risk mitigated by preserved oxygenation status.
