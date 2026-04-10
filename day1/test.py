import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 전처리 (통합본)
# ────────────────────────────────────────────────────────────
BASE = '/Users/rim/Desktop/workspace/project_1'

# Churn용/LTV용 데이터 로드
# (train_p.csv에 target_churn과 target_ltv가 모두 있다고 가정)
train = pd.read_csv(f'{BASE}/train_p.csv')
test = pd.read_csv(f'{BASE}/test_p.csv')

# 피처 정의
CHURN_FEATS = ['total_deposit_balance', 'card_loan_amt', 'credit_score', 'net_asset', 'days_since_joined']
LTV_FEATS = ['amt_sum', 'amt_mean', 'amt_max', 'total_deposit_balance', 'active_months', 'days_since_joined']

# ────────────────────────────────────────────────────────────
# 2. [PART 1] 이탈 예측 모델 (Classification)
# ────────────────────────────────────────────────────────────
print("\n[STEP 1] 이탈 예측(Churn) 모델 학습 시작...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
churn_oof = np.zeros(len(train))

churn_params = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'learning_rate': 0.014369, 'num_leaves': 31, 'n_estimators': 1000, 'random_state': 42
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, train['target_churn'])):
    X_tr, X_val = train[CHURN_FEATS].iloc[tr_idx], train[CHURN_FEATS].iloc[val_idx]
    y_tr, y_val = train['target_churn'].iloc[tr_idx], train['target_churn'].iloc[val_idx]

    m_churn = lgb.LGBMClassifier(**churn_params)
    m_churn.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    pred = m_churn.predict_proba(X_val)[:, 1]
    churn_oof[val_idx] = pred
    
    # Fold별 지표 계산
    auc = roc_auc_score(y_val, pred)
    mse = mean_squared_error(y_val, pred)
    rmse = np.sqrt(mse)
    fold_final_score = auc - (rmse / 2)

    print(f"========== Fold {fold+1} ==========")
    print(f"Fold {fold+1} AUC: {auc:.15f} "
          f"Fold {fold+1} RMSE: {rmse:.15f} "
          f"Fold {fold+1} MSE: {mse:.15f} "
          f"Fold {fold+1} Final Score: {fold_final_score:.15f}")

final_auc = roc_auc_score(train['target_churn'], churn_oof)

# ────────────────────────────────────────────────────────────
# 3. [PART 2] LTV 예측 모델 (Regression)
# ────────────────────────────────────────────────────────────
print("\n[STEP 2] LTV 예측 모델 학습 및 OOF RMSE 산출...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
ltv_oof = np.zeros(len(train))
y_ltv_log = np.log1p(train['target_ltv'])

for tr_idx, val_idx in kf.split(train):
    X_tr, X_val = train[LTV_FEATS].iloc[tr_idx], train[LTV_FEATS].iloc[val_idx]
    y_tr, y_val = y_ltv_log.iloc[tr_idx], y_ltv_log.iloc[val_idx]

    m_ltv = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=63, random_state=42, verbosity=-1)
    m_ltv.fit(X_tr, y_tr)
    
    # 지수 복원으로 실제 금액 스케일 RMSE 계산
    ltv_oof[val_idx] = np.expm1(m_ltv.predict(X_val))

final_ltv_rmse = np.sqrt(mean_squared_error(train['target_ltv'], ltv_oof))

# ────────────────────────────────────────────────────────────
# 4. [PART 3] 최종 통합 Score 산출 (질문자님 산식)
# ────────────────────────────────────────────────────────────
# LTV 변환 점수 : 1 / (1 + log(RMSE))
ltv_score = 1 / (1 + np.log(final_ltv_rmse))

# 최종 합산 : (0.5 * AUC) + (0.5 * LTV Score)
total_final_score = (0.5 * final_auc) + (0.5 * ltv_score)

print("\n" + "="*65)
print(f" [ 최종 통합 Score 산출 결과 ]")
print(f" AUC (Churn)   : {final_auc:.4f}")
print(f" RMSE (LTV)    : {final_ltv_rmse:,.2f}")
print(f" LTV 변환 점수 : 1 / (1 + log({final_ltv_rmse:,.0f})) = {ltv_score:.4f}")
print(" ───────────────────────────────────────────────────────────────")
print(f" 🏆 최종 Score  : 0.5 × {final_auc:.4f} + 0.5 × {ltv_score:.4f} = {total_final_score:.4f}")
print("="*65)

# ────────────────────────────────────────────────────────────
# 5. 제출 파일 생성
# ────────────────────────────────────────────────────────────
# (필요 시 테스트 데이터 예측 로직 추가 가능)
print("\n✅ 모든 계산이 완료되었습니다.")