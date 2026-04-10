import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score  # AUC 계산을 위해 추가

BASE = '/Users/rim/Desktop/workspace/project_1'
train = pd.read_csv(f'{BASE}/train_p.csv')
test = pd.read_csv(f'{BASE}/test_p.csv')

# 피처 리스트
CH_FEATS = ['fin_distress_v2', 'fin_overdue_days', 'card_loan_amt', 'fin_asset_trend_score', 
            'debt_to_deposit', 'total_deposit_balance', 'recency', 'trans_count', 
            'amt_m12', 'cnt_m12', 'credit_rank_in_income', 'days_since_joined']

# 지역별/카테고리별/소득별 평균 이탈률 정보 (Target Encoding)
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

# 모델 학습 및 AUC 계산
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(test))
auc_list = []  # 각 폴드별 AUC 저장용

for tr, val in skf.split(train[CH_FEATS], train['target_churn']):
    # 데이터 분할
    x_tr, y_tr = train[CH_FEATS].iloc[tr], train['target_churn'].iloc[tr]
    x_val, y_val = train[CH_FEATS].iloc[val], train['target_churn'].iloc[val]
    
    # 모델 학습 (기존 파라미터 유지)
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.014369, num_leaves=31, random_state=42, verbosity=-1)
    m.fit(x_tr, y_tr)
    
    # 검증셋 AUC 계산 및 저장
    val_prob = m.predict_proba(x_val)[:, 1]
    auc_list.append(roc_auc_score(y_val, val_prob))
    
    # 테스트셋 예측
    preds += m.predict_proba(test[CH_FEATS])[:, 1] / 5

# 결과 출력
print(f"\n✅ 각 폴드별 AUC: {auc_list}")
print(f"✅ 평균 AUC (Churn): {np.mean(auc_list):.4f}")

test['target_churn'] = preds
test[['customer_id', 'target_churn']].to_csv(f'{BASE}/pred_churn.csv', index=False)
print("✅ pred_churn.csv 생성 완료")