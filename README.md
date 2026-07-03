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
| Logistic Regression | 0.8725 | 0.4163 | 0.9109 | 0.5714 | 0.9509 |
| Random Forest | 0.9085 | 0.5063 | 0.7921 | 0.6178 | 0.9472 |
| XGBoost | 0.9076 | 0.5029 | 0.8515 | 0.6324 | 0.9550 |
| **Stacking Classifier (Best)** | **0.9298** | **0.6068** | **0.7030** | **0.6514** | **0.9511** |

The Stacking Classifier was selected as the best model based on the highest **F1 score (0.6514)** and **accuracy (92.98%)**, offering the best balance between precision and recall. While XGBoost had a marginally higher ROC-AUC, the Stacking Classifier's superior precision-recall balance made it more reliable for practical fraud-flagging decisions.

At the optimized decision threshold (0.338 instead of the default 0.5), recall improved to **0.762**, meaning the model catches more true fraud cases at a small, acceptable cost to precision.

**Top features driving fraud predictions:**
1. Total reimbursed amount
2. Total deductible
3. Inpatient claim count
4. Inpatient-to-outpatient claim ratio
5. Max claim amount

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
