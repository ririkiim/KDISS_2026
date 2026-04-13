# Repository for the 2026 KDISS Data Analysis Competition. 

<a href="https://kdiss.or.kr/board/competition_info/article/274389" target="_blank">
  <img src="https://github.com/user-attachments/assets/08095021-aa64-4ab1-b8ef-73cd550893dc" width="300" alt="KDISS Competition Banner">
</a>


#### 🚀 Update : v8 (2026-04-11)

##### 📊 v8 Summary (Current)
- **Churn AUC** : 0.7926 (Gap 0.05) 🟢
- **LTV RMSE** : 1.35M (Stagnant) 🔴
- **Total Score** : 0.4664
- **Key Discovery**: `card_loan` was a "mirage" signal; `Age` is the real driver for Churn=0.

##### 🔍 Key Issues
- **Stage-B Failure**: No predictive signal in VIP subset (Best Iter 0).
- **Target Noise**: Churn=1 (LTV 70k) ruins Churn=0 (LTV 1.3M) regression.

##### 🚀 v9 Action Plan
1. **Data Split**: Train LTV **ONLY** for Churn=0; set fixed LTV for Churn=1.
2. **Strategy Shift**: Pivot to **Classification-based Weighting** (5-bin LTV).
3. **Feature**: Intensive **Age Interactions** (Age × Asset, Age × Spend).
4. **Ensemble**: Combine **LGBM + XGBoost** for Churn AUC 0.81+.