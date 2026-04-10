import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error # RMSE 계산을 위해 추가

BASE = '/Users/rim/Desktop/workspace/project_1'
train = pd.read_csv(f'{BASE}/train_p.csv')
test = pd.read_csv(f'{BASE}/test_p.csv')

# 피처 리스트
LTV_FEATS = ['amt_sum', 'amt_mean', 'amt_max', 'total_deposit_balance', 'net_asset', 
             'active_months', 'amt_first_half', 'amt_second_half', 'half_growth_ratio', 
             'spend_rank_in_region', 'age', 'days_since_joined']

# 로그 변환
y_log = np.log1p(train['target_ltv']) 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(test))
rmse_list = [] # 각 폴드의 RMSE를 저장할 리스트

print("🚀 LTV 모델 학습 및 RMSE 계산 시작...")

for i, (tr, val) in enumerate(kf.split(train[LTV_FEATS])):
    # 학습/검증 데이터 분할
    x_tr, y_tr = train[LTV_FEATS].iloc[tr], y_log.iloc[tr]
    x_val, y_val = train[LTV_FEATS].iloc[val], y_log.iloc[val]
    
    # 모델 학습
    m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=63, random_state=42, verbosity=-1)
    m.fit(x_tr, y_tr)
    
    # 검증셋 예측 (로그 상태에서 예측)
    val_pred_log = m.predict(x_val)
    
    # 실제값과 예측값 모두 지수 복원 (원래 단위인 원/달러 등으로 복구)
    val_pred_real = np.expm1(val_pred_log)
    val_y_real = np.expm1(y_val)
    
    # RMSE 계산 (squared=False 옵션이 RMSE를 바로 반환합니다)
    rmse = mean_squared_error(val_y_real, val_pred_real, squared=False)
    rmse_list.append(rmse)
    
    print(f"  - Fold {i+1} RMSE: {rmse:,.2f}")
    
    # 테스트셋 예측 및 지수 복원 후 누적
    preds += np.expm1(m.predict(test[LTV_FEATS])) / 5

# 결과 출력
print("\n" + "="*30)
print(f"✅ 평균 RMSE (LTV): {np.mean(rmse_list):,.2f}")
print("="*30)

test['target_ltv'] = np.maximum(preds, 0)
test[['customer_id', 'target_ltv']].to_csv(f'{BASE}/pred_ltv.csv', index=False)
print("✅ pred_ltv.csv 생성 완료")