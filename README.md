# Healthcare Provider Fraud Detection

A machine learning project to identify potentially fraudulent healthcare providers using Medicare claims data, built as part of a Data Analytics internship.

## Problem Statement

Healthcare fraud costs billions annually and often hides in patterns across a provider's claims rather than in any single transaction. This project builds a classification model to flag providers likely to be committing fraud, based on aggregated billing behavior across their inpatient and outpatient claims.

## Dataset

- Beneficiary details, inpatient claims, and outpatient claims data, combined at the **provider level**
- 5,410 unique providers after preprocessing
- 33 engineered features
- Class imbalance: only ~9.4% of providers are labeled fraudulent

## Approach

1. **Data Merging** — Combined inpatient and outpatient claims with a claim-type flag, then aggregated to provider level.
2. **Feature Engineering** — Created features including:
   - Claims per patient
   - Reimbursed amount per claim / per patient
   - Deductible ratio
   - Inpatient-to-outpatient claim ratio
   - Max claim amount
3. **Preprocessing** — Handled missing values, scaled features with `StandardScaler`, and addressed class imbalance using **SMOTE**.
4. **Model Training** — Trained and compared four models using 5-fold stratified cross-validation:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Stacking Classifier (ensemble of the above)
5. **Threshold Optimization** — Tuned the classification threshold using the precision-recall curve to prioritize catching fraud cases (recall) over minimizing false alarms, since missed fraud is more costly than a false positive.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | – | – | – | 0.552 (CV) | – |
| Random Forest | – | – | – | 0.595 (CV) | – |
| XGBoost | 0.908 | 0.503 | – | 0.582 (CV) | – |
| **Stacking Classifier (Best)** | **0.930** | **0.607** | **0.703** | **0.651** | **0.951** |

At the optimized decision threshold (0.338 instead of the default 0.5), recall improved to **0.762**, meaning the model catches more true fraud cases at a small, acceptable cost to precision.

**Top features driving fraud predictions:**
1. Total reimbursed amount
2. Total deductible
3. Inpatient claim count
4. Inpatient-to-outpatient claim ratio
5. Max claim amount

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `Matplotlib` · `Seaborn`

## Project Structure
├── ML_Project.ipynb      # Full notebook: data prep, feature engineering, modeling, evaluation
├── models/                # Saved trained models
└── README.md

## Key Takeaway

Provider-level aggregation combined with ratio-based features (rather than raw claim counts) was key to separating fraudulent billing patterns from legitimate ones. The stacking ensemble outperformed any single model, and threshold tuning meaningfully improved fraud recall — the metric that matters most in this use case.
