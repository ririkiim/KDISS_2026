# 2026 KDISS & SAS Data Analysis Competition

## 1. Overview
This project focuses on predicting customer behavior by integrating demographics, transaction history, and financial profiles.

- **Churn** : 1-month exit probability (ROC-AUC)
- **LTV** : 1-year virtual Lifetime Value (RMSE)

---

## 2. Performance
> Score Formula: $0.5 \times AUC(\text{Churn}) + 0.5 \times \frac{1}{1 + \log(RMSE_{\text{LTV}})}$


#### Final Results
The final model improved upon the baseline (0.42987) by incorporating **dynamic trend features** and **model ensembling**.

| Metric | Score | Remarks |
| :--- | :---: | :--- |
| **Integrated Score** | **0.43172** | **+0.00185 improvement** |
| **Churn AUC** | **0.7974** | Consistent across folds |
| **LTV RMSE** | **1,384,881** | Stabilized via transformation |

#### 5-Fold Cross-Validation
Performance remained stable across all folds, indicating good generalization.
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

The focus shifted from simple aggregates to capturing **behavioral changes** and **financial risk signals**.
- **Top Feature : `net_cash_flow`**
  
  The most important predictor. A customer’s net liquidity proved to be more informative than total spending alone.
- **Volatility (`amt_cv`)**

  A top contributor that captured a distinct “high-spender churn” segment—customers with large but inconsistent transactions.
- **Spending Trends (`amt_trend_ratio`)**
  
  Designed to detect recent drops in activity compared to longer-term behavior.
- **Financial Stress Features**
  
  `loan_asset_ratio` and `debt_pressure` were introduced to reflect financial instability.
- **Data Consistency**
  
  Fixed LabelEncoder scope issues and aligned feature ordering between training and inference.

---

## 4. Modeling Strategy
- **Ensemble (LGBM + CatBoost)**
  
  Combined LightGBM’s efficiency with CatBoost’s strength in handling categorical variables, resulting in a +0.00185 improvement over single models.
- **Target Optimization**
  
  Applied **square root transformation** to LTV to reduce skewness. Post-processing included clipping negative predictions.
- **Categorical Handling**
  
  Resolved LabelEncoder inconsistencies to ensure stable behavior during inference.

---

## 5. Model Evolution
Performance improvements were driven more by **feature design** than by hyperparameter tuning.

| Model Architecture | Key Enhancements | Score |
| :--- | :--- | :---: |
| Baseline (LGBM) | Basic RFM + Raw balances | 0.42987 |
| **Final Ensemble** | **Trend features + Ensemble (LGBM + CatBoost) + Sqrt Transform** | **0.43172** |

---

## 6. Insights
- **Cash Flow Matters More Than Volume**
  
  Even high-income customers showed high churn risk when `net_cash_flow` turned negative.
- **Consistency Over Scale**
  
  Customers with stable spending patterns were more likely to stay than those with high but volatile activity.
- **Handling LTV Skewness**
  
  A small number of high-value customers dominated the distribution. Applying transformation and focusing on trends improved stability.

---

## Quick Start
```bash
git clone [https://github.com/rim/KDISS-2026.git](https://github.com/rim/KDISS-2026.git)
