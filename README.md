# 2026 KDISS Data Analysis Competition. 

<a href="https://kdiss.or.kr/board/competition_info/article/274389" target="_blank">
  <img src="https://github.com/user-attachments/assets/08095021-aa64-4ab1-b8ef-73cd550893dc" width="300" alt="KDISS Competition Banner">
</a>

### v9 Report (2026-04-22)

##### 📊 Performance
- **Total Score**: `0.42987`
- **Churn AUC**: `0.7937` 
- **LTV RMSE**: `1.38M`

##### 🛠 Updates
- **Features**: `tx_density`, `amt_cv`, `loan_asset_ratio`
- **Tuning**: `LR 0.002`, `Leaves 127`, `Extra Trees`
- **Stability**: `Gap < 0.05` (Overfit Control, Stable except Fold 2)

<details>
<summary></summary>
</details>

##### 📈 Importance
- **Churn**: `Assets`, `Loans`, `Credit`
- **LTV**: `Region`, `Category`, **`amt_cv`**, **`tx_density`**

---


<details>
<summary> Update : v8 (2026-04-11)</summary>

##### 📊 v8 Summary (Current)
- **Churn AUC** : 0.7926 (Gap 0.05) 🟢
- **LTV RMSE** : 1.35M (Stagnant) 🔴
- **Total Score** : 0.4309
- **Key Discovery**: `card_loan` was a "mirage" signal; `Age` is the real driver for Churn=0.

##### 🔍 Key Issues
- **Stage-B Failure**: No predictive signal in VIP subset (Best Iter 0).
- **Target Noise**: Churn=1 (LTV 70k) ruins Churn=0 (LTV 1.3M) regression.

</details>

