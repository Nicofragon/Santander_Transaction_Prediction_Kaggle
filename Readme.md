# Santander Customer Transaction Prediction (Kaggle)

Binary classification on Santander's anonymized banking dataset: predict which
customers will make a specific transaction in the future. Gradient-boosted trees
(LightGBM) on 200 anonymized features, optimized for ROC-AUC.

## Approach
- **Data:** 200,000 rows × 200 anonymized numeric features (`var_0`…`var_199`);
  heavily imbalanced target (≈90% / 10%).
- **EDA:** null check, target balance, per-feature distributions and correlations.
- **Baseline:** `SGDClassifier` (log loss) on standardized features, tuned with
  `RandomizedSearchCV` (CV ROC-AUC ≈ 0.86).
- **Model:** `LightGBM` (`LGBMClassifier`, gradient boosting), hyper-parameters
  tuned with Bayesian optimization (`BayesSearchCV`, 3-fold stratified CV).
- **Validation:** 70/30 train/validation hold-out, metric = ROC-AUC.

## Results
- Tuned LightGBM: **validation ROC-AUC ≈ 0.91** (default LightGBM ≈ 0.885).
- Generates `lgb_submission.csv` for the Kaggle leaderboard.
- *(No final leaderboard rank is recorded in this repo — the reported score is
  the local validation ROC-AUC.)*

## Stack
Python · pandas · NumPy · scikit-learn · LightGBM · scikit-optimize · seaborn/matplotlib

> _(2021–22, Kaggle-era project — my production work since lives in private client repos.)_

Challenge: https://www.kaggle.com/competitions/santander-customer-transaction-prediction
