import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 경로 및 파라미터 설정
# ==========================================
BASE = '/Users/rim/Desktop/workspace/project_1/KDISS-2026/2회 경진대회 데이터'

# [Churn] 
CHURN_PARAMS = {
    'n_estimators': 10000,
    'learning_rate': 0.002,
    'num_leaves': 48,
    'min_child_samples': 40,
    'scale_pos_weight': 3.5,
    'subsample': 0.8,
    'colsample_bytree': 0.8, 
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1
}

# [LTV] 
LTV_PARAMS = {
    'n_estimators': 5000,
    'learning_rate': 0.002,
    'num_leaves': 127,     
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 5.0,
    'extra_trees': True,     
    'random_state': 42,
    'verbose': -1
}

# ==========================================
# 2. 전처리 함수 (Train/Test 공통 적용)
# ==========================================
def get_features(cust_path, tran_path, fin_path, target_df=None):
    cust = pd.read_csv(cust_path)
    tran = pd.read_csv(tran_path)
    fin = pd.read_csv(fin_path)
    
    ref_date = pd.Timestamp('2024-01-01')
    tran['trans_date'] = pd.to_datetime(tran['trans_date'])
    
  
    agg = tran.groupby('customer_id').agg(
        recency=('trans_date', lambda x: (ref_date - x.max()).days),
        frequency=('trans_id', 'count'),
        amt_sum=('trans_amount', 'sum'),
        amt_mean=('trans_amount', 'mean'),
        amt_std=('trans_amount', 'std'),
        amt_max=('trans_amount', 'max')
    ).reset_index()

    # [추가 1] 거래 금액 변동성 (CV)
    agg['amt_cv'] = agg['amt_std'] / (agg['amt_mean'] + 1)
    
    # [추가 2] 최근 트렌드
    t_l1m = tran[tran['trans_date'] >= pd.Timestamp('2023-12-01')]
    t_p5m = tran[tran['trans_date'] <  pd.Timestamp('2023-12-01')]
    
    l1m_sum = t_l1m.groupby('customer_id')['trans_amount'].sum().rename('amt_l1m')
    p5m_sum = t_p5m.groupby('customer_id')['trans_amount'].sum().rename('amt_p5m')
    
    trend = pd.merge(l1m_sum, p5m_sum, on='customer_id', how='outer').fillna(0)
    trend['amt_trend_ratio'] = (trend['amt_l1m'] - trend['amt_p5m']/5) / (trend['amt_p5m']/5 + 1)
    
    # [추가 3] 기타 패턴
    gap = tran.groupby('customer_id')['trans_date'].apply(lambda x: x.sort_values().diff().dt.days.mean()).rename('mean_gap')
    onl = tran.groupby('customer_id').apply(lambda x: (x['biz_type'] == 'Online').mean()).rename('online_ratio')
    ins = tran.groupby('customer_id')['is_installment'].mean().rename('install_ratio')
    
    # 병합
    feats = agg.merge(trend, on='customer_id', how='left')\
               .merge(gap, on='customer_id', how='left')\
               .merge(onl, on='customer_id', how='left')\
               .merge(ins, on='customer_id', how='left')
    
    # 전체 데이터셋 구성
    if target_df is not None:
        df = target_df.merge(cust, on='customer_id', how='left')
    else:
        df = cust.copy()
        
    df = df.merge(fin, on='customer_id', how='left').merge(feats, on='customer_id', how='left')
    
    # [날짜 변환]
    df['join_date'] = pd.to_datetime(df['join_date'])
    df['join_days'] = (ref_date - df['join_date']).dt.days
    
    # [추가 4] 거래 밀도 (가입 기간 대비 얼마나 자주?)
    df['tx_density'] = df['frequency'] / (df['join_days'] + 1)
    
    # [추가 5] 대출/예금 비율
    df['loan_asset_ratio'] = df['total_loan_balance'] / (df['total_deposit_balance'] + 1)
    
    return df.drop(columns=['join_date']).fillna(0)

# ==========================================
# 3. 데이터 로드 및 피처 생성
# ==========================================
print("--- [1] 데이터 로드 및 전처리 시작 ---")
train_target = pd.read_csv(f'{BASE}/train/train_targets.csv')
train_df = get_features(f'{BASE}/train/train_customer_info.csv', 
                        f'{BASE}/train/train_transaction_history.csv', 
                        f'{BASE}/train/train_finance_profile.csv', 
                        train_target)

test_df = get_features(f'{BASE}/test/test_customer_info.csv', 
                       f'{BASE}/test/test_transaction_history.csv', 
                       f'{BASE}/test/test_finance_profile.csv')

# 인코딩
cat_cols = ['gender', 'region_code', 'prefer_category', 'income_group']
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = test_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# X, y 분리
X = train_df.drop(columns=['customer_id', 'target_churn', 'target_ltv'])
y_churn = train_df['target_churn']
y_ltv_sqrt = np.sqrt(train_df['target_ltv'])
X_test = test_df[X.columns]

# ==========================================
# 4. K-Fold 학습 및 검증
# ==========================================
print("--- [2] K-Fold 학습 및 검증 시작 ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, rmse_scores = [], []
churn_preds, ltv_preds = np.zeros(len(X_test)), np.zeros(len(X_test))

for fold, (t_idx, v_idx) in enumerate(skf.split(X, y_churn)):
    X_tr, X_vl = X.iloc[t_idx], X.iloc[v_idx]
    yc_tr, yc_vl = y_churn.iloc[t_idx], y_churn.iloc[v_idx]
    yl_tr, yl_vl = y_ltv_sqrt.iloc[t_idx], y_ltv_sqrt.iloc[v_idx]

    # Churn
    m_c = lgb.LGBMClassifier(**CHURN_PARAMS)
    m_c.fit(X_tr, yc_tr, eval_set=[(X_vl, yc_vl)], 
            callbacks=[lgb.early_stopping(300), lgb.log_evaluation(-1)])
    fold_auc = roc_auc_score(yc_vl, m_c.predict_proba(X_vl)[:, 1])
    # print(f"Fold {fold+1} Overfit Check - Train AUC: {roc_auc_score(yc_tr, m_c.predict_proba(X_tr)[:, 1]):.4f} / Val AUC: {fold_auc:.4f}")
    auc_scores.append(fold_auc)
    churn_preds += m_c.predict_proba(X_test)[:, 1] / 5

    # LTV
    m_l = lgb.LGBMRegressor(**LTV_PARAMS)
    m_l.fit(X_tr, yl_tr, eval_set=[(X_vl, yl_vl)], 
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(-1)])
    vl_ltv_pred = np.maximum(m_l.predict(X_vl), 0) ** 2
    fold_rmse = np.sqrt(mean_squared_error(train_df['target_ltv'].iloc[v_idx], vl_ltv_pred))
    rmse_scores.append(fold_rmse)
    ltv_preds += (np.maximum(m_l.predict(X_test), 0) ** 2) / 5
    
    print(f"Fold {fold+1}: AUC = {fold_auc:.4f}, RMSE = {fold_rmse:,.0f}")

# ==========================================
# 5. 최종 결과 리포트
# ==========================================
mean_auc = np.mean(auc_scores)
mean_rmse = np.mean(rmse_scores)
final_score = 0.5 * mean_auc + 0.5 * (1 / (1 + np.log(mean_rmse)))

print(f"\n{'='*45}")
print(f"CV AUC  평균: {mean_auc:.4f}")
print(f"CV RMSE 평균: {mean_rmse:,.0f}")
print(f"대회 산식 최종 Score: {final_score:.5f}")
print(f"{'='*45}")

# submission = pd.DataFrame({
#     'customer_id': test_df['customer_id'],
#     'target_churn': churn_preds,
#     'target_ltv': ltv_preds
# })
# submission.to_csv('submission_v9.csv', index=False)