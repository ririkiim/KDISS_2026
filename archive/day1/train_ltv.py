import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

BASE = '/Users/rim/Desktop/workspace/project_1'
train = pd.read_csv(f'{BASE}/train_p.csv')
test = pd.read_csv(f'{BASE}/test_p.csv')

# amt_sum(총결제액), amt_mean(평균액), amt_max(최대액), active_months(활동달수),
# amt_first_half(전반기금액), half_growth_ratio(성장률), spend_rank_in_region(지역내순위) 등 총 44개 사용
LTV_FEATS = ['amt_sum', 'amt_mean', 'amt_max', 'total_deposit_balance', 'net_asset', 
             'active_months', 'amt_first_half', 'amt_second_half', 'half_growth_ratio', 
             'spend_rank_in_region', 'age', 'days_since_joined']

# 로그 변환
y_log = np.log1p(train['target_ltv']) 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(test))
for tr, val in kf.split(train[LTV_FEATS]):
    # LTV 베스트 파라미터 적용 (learning_rate: 0.01, num_leaves: 63)
    m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=63, random_state=42, verbosity=-1)
    m.fit(train[LTV_FEATS].iloc[tr], y_log.iloc[tr])
    # 지수 복원
    preds += np.expm1(m.predict(test[LTV_FEATS])) / 5

test['target_ltv'] = np.maximum(preds, 0)
test[['customer_id', 'target_ltv']].to_csv(f'{BASE}/pred_ltv.csv', index=False)
print("✅ pred_ltv.csv 생성 완료")