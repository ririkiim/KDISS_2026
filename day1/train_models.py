import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold

BASE = '/Users/rim/Desktop/workspace/project_1'
train = pd.read_csv(f'{BASE}/train_p.csv')
test = pd.read_csv(f'{BASE}/test_p.csv')

# fin_distress_v2(위기지표), fin_overdue_days(연체), card_loan_amt(카드론), 
# total_deposit_balance(예금), recency(마지막결제일), trans_count(거래수), 
# amt_m12(최근달결제액), days_since_joined(가입기간) 등 총 31개 사용

CH_FEATS = ['fin_distress_v2', 'fin_overdue_days', 'card_loan_amt', 'fin_asset_trend_score', 
            'debt_to_deposit', 'total_deposit_balance', 'recency', 'trans_count', 
            'amt_m12', 'cnt_m12', 'credit_rank_in_income', 'days_since_joined']

# 지역별/카테고리별/소득별 평균 이탈률 정보
for c in ['region_code', 'prefer_category', 'income_group']:
    f = f'{c}_churn_rate'
    train[f] = 0.0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr, val in kf.split(train):
        m = train.iloc[tr].groupby(c)['target_churn'].mean()
        train.loc[train.index[val], f] = train.loc[train.index[val], c].map(m)
    train[f].fillna(train['target_churn'].mean(), inplace=True)
    test[f] = test[c].map(train.groupby(c)['target_churn'].mean()).fillna(train['target_churn'].mean())
    CH_FEATS.append(f)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(test))
for tr, val in skf.split(train[CH_FEATS], train['target_churn']):
    # Optuna 베스트 파라미터 적용
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.014369, num_leaves=31, random_state=42, verbosity=-1)
    m.fit(train[CH_FEATS].iloc[tr], train['target_churn'].iloc[tr])
    preds += m.predict_proba(test[CH_FEATS])[:, 1] / 5

test['target_churn'] = preds
test[['customer_id', 'target_churn']].to_csv(f'{BASE}/pred_churn.csv', index=False)
print("✅ pred_churn.csv 생성 완료")