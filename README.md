# 2026 KDISS & SAS KOREA Data Analysis Competition

## 1. Overview
This project predicts customer behavior by integrating demographics, transaction history, and financial profiles. The model focuses on two primary objectives:
- **Churn**: 1-month exit probability (Optimized via ROC-AUC)
- **LTV**: 1-year virtual Lifetime Value (Optimized via RMSE)

---

## 2. Performance

> **Score Formula**: $0.5 \times AUC(\text{Churn}) + 0.5 \times \frac{1}{1 + \log(RMSE_{\text{LTV}})}$
> 
*Note: the LTV term approaches 0.5 as RMSE approaches 0*

#### Final Results
The final ensemble model improved the baseline score of 0.42987 by incorporating **dynamic trend features** and **model ensembling**.

| Metric | Score | Remarks |
| :--- | :---: | :--- |
| **Integrated Score** | **0.43172** | **+0.00185 improvement from baseline** |
| **Churn AUC** | **0.7974** | Stable generalization across 5-folds |
| **LTV RMSE** | **1,384,881** | Stabilized via Target Transformation |

#### 5-Fold Cross-Validation
Results were stable across folds with no significant outliers.
| Fold | Churn AUC | LTV RMSE |
| :---: | :---: | :---: |
| 1 | 0.7972 | 1,382,460 |
| 2 | 0.7974 | 1,409,119 |
| 3 | 0.8018 | 1,388,259 |
| 4 | 0.7922 | 1,347,404 |
| 5 | 0.7986 | 1,395,172 |
| **Mean** | **0.7974** | **1,384,483** |

---

## 3. Key Feature Engineering
The focus was shifted from simple spending aggregates to capturing **behavioral shifts** and **financial risk signals**.

- **Net Cash Flow (Primary Predictor)**: A customer’s net liquidity turned out to be a stronger predictor than total spending in predicting churn.
- **Volatility (`amt_cv`)**: Used the Coefficient of Variation to identify "high-spender churn" segments—customers with large but irregular transaction patterns.
- **Spending Trends (`amt_trend_ratio`)**: Captures cases where activity in the latest month dropped compared to the 5-month moving average.
- **Financial Stress**: Introduced `loan_asset_ratio` and `debt_pressure` to capture household financial stress linked to higher churn risk.

---

## 4. Modeling Strategy & Troubleshooting

#### Modeling Strategy
- **Ensemble (LGBM + CatBoost)**: Used a 50/50 blend of LightGBM and CatBoost — LightGBM     for efficiency and CatBoost for categorical feature handling.
  The 50/50 blend ratio and key hyperparameters were selected based on Optuna trials       ([`src/optuna_tuning.py`](./src/optuna_tuning.py)). 
- **Target Optimization**: Applied a **Square Root (Sqrt) Transformation** to LTV to reduce skewness. Negative predictions were clipped to 0 after inference.

#### Lessons Learned & Troubleshooting
- **LabelEncoder Consistency**: Resolved a critical issue where the encoder scope differed between training and inference sets, which previously caused inconsistent predictions.
- **Feature Alignment**: Synchronized feature ordering between the training pipeline and the inference script to prevent data distortion during real-time prediction.

---

## 5. Model Evolution
Performance was primarily driven by **feature engineering** rather than extensive hyperparameter tuning.

| Model Architecture | Key Enhancements | Score |
| :--- | :--- | :---: |
| Baseline (LGBM) | Basic RFM + Raw balances | 0.42987 |
| **Final Ensemble** | **Trend features + LGBM/CatBoost Blend + Sqrt Transform** | **0.43172** |

---

## 6. Insights

- **Cash Flow Thresholds**: Customers whose `net_cash_flow` turned negative had an **attrition rate 2.4x higher** than those with positive liquidity.
- **Volatility as a Risk Signal**: Customers in the top 10% of spending volatility (`amt_cv`) were **1.8x more likely to churn**, even if their total asset volume remained high.
- **LTV Skewness Control**: We found that less than 5% of high-value outliers accounted for over 60% of total RMSE. Target scaling and trend-based features reduced the impact of these outliers on overall RMSE.

---

## Quick Start

### ⚙️ Environment
- **Python**: 3.9+
- **Key Dependencies**: `lightgbm`, `catboost`, `optuna`, `pandas`, `scikit-learn`

### 📂 Execution
1. **Clone Repository**: `git clone https://github.com/ririkiim/KDISS_2026.git`
2. (Optional) Reproduce hyperparameter search: `python src/optuna_tuning.py`
3. **Generate Submission**: `python src/train.py`
   
*For detailed experimental logs and daily trial scripts, please refer to the [`/archive`](./archive) directory.*

---
