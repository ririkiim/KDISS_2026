import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────────────────────
BASE = '/Users/rim/Desktop/workspace/project_1'
train = pd.read_csv(f'{BASE}/train_p.csv')
test = pd.read_csv(f'{BASE}/test_p.csv')

# ────────────────────────────────────────────────────────────
# 2. 피처 엔지니어링 (기존 성능을 냈던 피처 리스트 구성)
# ────────────────────────────────────────────────────────────
# Churn용 피처 (AUC 0.80을 기록했던 핵심 변수들)
CH_FEATS = ['fin_distress_v2', 'fin_overdue_days', 'card_loan_amt', 'fin_asset_trend_score', 
            'debt_to_deposit', 'total_deposit_balance', 'recency', 'trans_count', 
            'amt_m12', 'cnt_m12', 'credit_rank_in_income', 'days_since_joined']

# 타겟 인코딩 (데이터 누수 방지를 위한 OOF 방식 적용)
for c in ['region_code', 'prefer_category', 'income_group']:
    f = f'{c}_churn_rate'
    train[f] = 0.0
    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf_te.split(train):
        mapping = train.iloc[tr_idx].groupby(c)['target_churn'].mean()
        train.loc[train.index[val_idx], f] = train.loc[train.index[val_idx], c].map(mapping)
    train[f].fillna(train['target_churn'].mean(), inplace=True)
    # Test 데이터는 Train 전체 평균으로 매핑
    test[f] = test[c].map(train.groupby(c)['target_churn'].mean()).fillna(train['target_churn'].mean())
    CH_FEATS.append(f)

# LTV용 피처
LTV_FEATS = ['amt_sum', 'amt_mean', 'amt_max', 'total_deposit_balance', 'net_asset', 
             'active_months', 'amt_first_half', 'amt_second_half', 'half_growth_ratio', 
             'spend_rank_in_region', 'age', 'days_since_joined']

# ────────────────────────────────────────────────────────────
# 3. [PART 1] Churn 모델 (이탈 예측) - AUC 0.80 복원
# ────────────────────────────────────────────────────────────
print("\n[STEP 1] 이탈 예측(Churn) 모델 학습 시작...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
churn_oof = np.zeros(len(train))
churn_test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train[CH_FEATS], train['target_churn'])):
    X_tr, X_val = train[CH_FEATS].iloc[tr_idx], train[CH_FEATS].iloc[val_idx]
    y_tr, y_val = train['target_churn'].iloc[tr_idx], train['target_churn'].iloc[val_idx]

    # 파라미터 재점검 (AUC 극대화 설정)
    m_ch = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.014369, num_leaves=31, 
                               random_state=42, verbosity=-1, importance_type='gain')
    
    m_ch.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

    val_pred = m_ch.predict_proba(X_val)[:, 1]
    churn_oof[val_idx] = val_pred
    churn_test_preds += m_ch.predict_proba(test[CH_FEATS])[:, 1] / 5

    # 지표 산출
    auc = roc_auc_score(y_val, val_pred)
    mse = mean_squared_error(y_val, val_pred)
    rmse = np.sqrt(mse)
    # 여기서 Final Score는 Churn 모델 내부 지표 (AUC - RMSE/2)
    fold_score = auc - (rmse / 2)

    print(f"========== Fold {fold+1} ==========")
    print(f"Fold {fold+1} AUC: {auc:.15f} Fold {fold+1} RMSE: {rmse:.15f} Fold {fold+1} MSE: {mse:.15f} Fold {fold+1} Final Score: {fold_score:.15f}")

final_auc = roc_auc_score(train['target_churn'], churn_oof)

# ────────────────────────────────────────────────────────────
# 4. [PART 2] LTV 모델 (가치 예측)
# ────────────────────────────────────────────────────────────
print("\n[STEP 2] LTV 예측 모델 학습 및 OOF RMSE 산출...")
kf_ltv = KFold(n_splits=5, shuffle=True, random_state=42)
ltv_oof = np.zeros(len(train))
ltv_test_preds = np.zeros(len(test))
y_ltv_log = np.log1p(train['target_ltv'])

for tr_idx, val_idx in kf_ltv.split(train[LTV_FEATS]):
    m_ltv = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=63, 
                               random_state=42, verbosity=-1)
    m_ltv.fit(train[LTV_FEATS].iloc[tr_idx], y_ltv_log.iloc[tr_idx])
    
    # 예측값 복원 및 저장
    val_pred_ltv = np.expm1(m_ltv.predict(train[LTV_FEATS].iloc[val_idx]))
    ltv_oof[val_idx] = val_pred_ltv
    ltv_test_preds += np.expm1(m_ltv.predict(test[LTV_FEATS])) / 5

final_ltv_rmse = np.sqrt(mean_squared_error(train['target_ltv'], ltv_oof))

# ────────────────────────────────────────────────────────────
# 5. [PART 3] 최종 통합 Score 산출 (가장 중요)
# ────────────────────────────────────────────────────────────
ltv_score = 1 / (1 + np.log(final_ltv_rmse))
total_final_score = (0.5 * final_auc) + (0.5 * ltv_score)

print("\n" + "="*65)
print(f" [ 최종 통합 Score 산출 ]")
print(f" AUC (Churn) : {final_auc:.4f}")
print(f" RMSE (LTV)  : {final_ltv_rmse:,.2f}")
print(f" LTV 변환 점수 : 1 / (1 + log({final_ltv_rmse:,.0f})) = {ltv_score:.4f}")
print(" ───────────────────────────────────────────────────────────────")
print(f" 최종 Score : 0.5 × {final_auc:.4f} + 0.5 × {ltv_score:.4f} = {total_final_score:.4f}")
print("="*65)

# ────────────────────────────────────────────────────────────
# 6. 제출 파일 저장
# ────────────────────────────────────────────────────────────
test['target_churn'] = churn_test_preds
test['target_ltv'] = np.maximum(ltv_test_preds, 0)
test[['customer_id', 'target_churn', 'target_ltv']].to_csv(f'{BASE}/final_submission.csv', index=False)
print(f"\n✅ 제출 파일 생성 완료: {BASE}/final_submission.csv")