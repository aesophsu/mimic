# Table S5. Optuna Search Space and Best Hyperparameters

## A. Hyperparameter Search Space

| Model | Hyperparameter | Type | Search Space | Sampling Distribution | Optuna Trials (per endpoint) |
|---|---|---|---|---|---|
| XGBoost | n_estimators | int | [100, 500] | uniform integer | 100 |
| XGBoost | max_depth | int | [3, 7] | uniform integer | 100 |
| XGBoost | learning_rate | float | [0.01, 0.2] | log-uniform | 100 |
| XGBoost | subsample | float | [0.5, 1.0] | uniform | 100 |
| Random Forest | n_estimators | int | [100, 500] | uniform integer | 100 |
| Random Forest | max_depth | int | [5, 15] | uniform integer | 100 |
| Random Forest | min_samples_split | int | [2, 20] | uniform integer | 100 |
| Random Forest | min_samples_leaf | int | [1, 10] | uniform integer | 100 |
| SVM | C | float | [0.1, 10.0] | log-uniform | 50 |
| SVM | gamma | categorical | {scale, auto} | categorical | 50 |
| SVM | kernel | fixed | rbf | fixed | 50 |
| Decision Tree | max_depth | int | [3, 15] | uniform integer | 50 |
| Decision Tree | min_samples_leaf | int | [1, 20] | uniform integer | 50 |

## B. Best Hyperparameter Combination

| Endpoint | Model | Best Hyperparameter Combination |
|---|---|---|
| **Primary endpoint (POF)** |  |  |
| Primary endpoint (POF) | XGBoost | `{"n_estimators": 203, "max_depth": 3, "learning_rate": 0.015356541710370732, "subsample": 0.7904316109837108}` |
| Primary endpoint (POF) | Random Forest | `{"n_estimators": 429, "max_depth": 14, "min_samples_split": 19, "min_samples_leaf": 4}` |
| Primary endpoint (POF) | SVM | `{"C": 0.22032188337418315, "gamma": "scale", "kernel": "rbf"}` |
| Primary endpoint (POF) | Decision Tree | `{"max_depth": 9, "min_samples_leaf": 18}` |
| **Secondary endpoint (28-day mortality)** |  |  |
| Secondary endpoint (28-day mortality) | XGBoost | `{"n_estimators": 211, "max_depth": 3, "learning_rate": 0.02602881259023273, "subsample": 0.5235190860568599}` |
| Secondary endpoint (28-day mortality) | Random Forest | `{"n_estimators": 372, "max_depth": 15, "min_samples_split": 4, "min_samples_leaf": 4}` |
| Secondary endpoint (28-day mortality) | SVM | `{"C": 0.10008101329804245, "gamma": "auto", "kernel": "rbf"}` |
| Secondary endpoint (28-day mortality) | Decision Tree | `{"max_depth": 6, "min_samples_leaf": 15}` |
| **Composite endpoint** |  |  |
| Composite endpoint | XGBoost | `{"n_estimators": 348, "max_depth": 3, "learning_rate": 0.012276155565382528, "subsample": 0.5021181667841529}` |
| Composite endpoint | Random Forest | `{"n_estimators": 421, "max_depth": 9, "min_samples_split": 2, "min_samples_leaf": 6}` |
| Composite endpoint | SVM | `{"C": 0.3339704119445903, "gamma": "scale", "kernel": "rbf"}` |
| Composite endpoint | Decision Tree | `{"max_depth": 8, "min_samples_leaf": 19}` |
