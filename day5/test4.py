import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 경로 및 파라미터 설정 (Optuna 최적화 값)
# ==========================================
BASE = '/Users/rim/Desktop/workspace/project_1/KDISS-2026/2회 경진대회 데이터'

CHURN_PARAMS = {
    'n_estimators': 10000,
    'learning_rate': 0.005,       
    'num_leaves': 31,
    'min_child_samples': 30,
    'scale_pos_weight': 3.378,
    'subsample': 0.857,
    'colsample_bytree': 0.570,
    'reg_alpha': 1.785,
    'reg_lambda': 0.318,
    'random_state': 42,
    'verbose': -1
}

LTV_PARAMS = {
    'n_estimators': 3000,
    'learning_rate': 0.003,
    'num_leaves': 63,
    'min_child_samples': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1
}


# Seed 설정
#SEED = 312

#CHURN_PARAMS['random_state'] = SEED
#LTV_PARAMS['random_state'] = SEED
#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ==========================================
# 2. 피처 생성 함수 
# ==========================================
def make_features(customer_df, finance_df, transaction_df, ref_date):
    df_tran = transaction_df.copy()
    df_tran['trans_date'] = pd.to_datetime(df_tran['trans_date'])
    
    agg = df_tran.groupby('customer_id').agg(
        recency=('trans_date', lambda x: (ref_date - x.max()).days),
        frequency=('trans_id', 'count'),
        total_amount=('trans_amount', 'sum'),
        mean_amount=('trans_amount', 'mean'),
        std_amount=('trans_amount', 'std'),
        max_amount=('trans_amount', 'max'),
        active_days=('trans_date', lambda x: x.dt.date.nunique()),
        cat_nunique=('item_category', 'nunique')
    ).reset_index()
    
    agg['amt_cv'] = agg['std_amount'] / (agg['mean_amount'] + 1)
    agg['tx_per_active_day'] = agg['frequency'] / (agg['active_days'] + 1)
    
    l1m_limit = ref_date - pd.DateOffset(months=1)
    l3m_limit = ref_date - pd.DateOffset(months=3)
    
    amt_l1m = df_tran[df_tran['trans_date'] >= l1m_limit].groupby('customer_id')['trans_amount'].sum().rename('amt_l1m')
    amt_l3m = df_tran[df_tran['trans_date'] >= l3m_limit].groupby('customer_id')['trans_amount'].sum().rename('amt_l3m')
    
    agg = agg.merge(amt_l1m, on='customer_id', how='left').merge(amt_l3m, on='customer_id', how='left').fillna(0)
    agg['amt_trend_ratio'] = (agg['amt_l1m'] - (agg['amt_l3m']/3)) / ((agg['amt_l3m']/3) + 1)

    df = customer_df.merge(finance_df, on='customer_id', how='left').merge(agg, on='customer_id', how='left')
    df['join_days'] = (ref_date - pd.to_datetime(df['join_date'])).dt.days
    df['net_cash_flow'] = df['total_deposit_balance'] - df['card_loan_amt']
    df['debt_pressure'] = df['total_loan_balance'] / (df['credit_score'] + 1)
    
    drop_cols = ['gender', 'is_married', 'high_amt_user', 'fin_overdue_days', 'join_date', 'region_code', 'loan_to_income']
    actual_drops = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=actual_drops).fillna(0)

# ==========================================
# 3. 데이터 로드 및 사전 준비
# ==========================================
train_cust = pd.read_csv(f'{BASE}/train/train_customer_info.csv')
train_tran = pd.read_csv(f'{BASE}/train/train_transaction_history.csv')
train_fin = pd.read_csv(f'{BASE}/train/train_finance_profile.csv')
train_target = pd.read_csv(f'{BASE}/train/train_targets.csv')

test_cust = pd.read_csv(f'{BASE}/test/test_customer_info.csv')
test_tran = pd.read_csv(f'{BASE}/test/test_transaction_history.csv')
test_fin = pd.read_csv(f'{BASE}/test/test_finance_profile.csv')

REF_DATE_TRAIN = pd.to_datetime(train_tran['trans_date']).max()
REF_DATE_TEST = pd.to_datetime(test_tran['trans_date']).max()

X_test_base = make_features(test_cust, test_fin, test_tran, REF_DATE_TEST)

train_target['ltv_bin'] = pd.qcut(train_target['target_ltv'], 5, labels=False, duplicates='drop')
train_target['stratify_col'] = train_target['target_churn'].astype(str) + "_" + train_target['ltv_bin'].astype(str)

# ==========================================
# 4. K-Fold 학습 및 검증 시작
# ==========================================
print("--- K-Fold 학습 및 검증 시작 ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, rmse_scores = [], []
churn_preds, ltv_preds = np.zeros(len(test_cust)), np.zeros(len(test_cust))

for fold, (t_idx, v_idx) in enumerate(skf.split(train_target, train_target['stratify_col'])):
    t_ids, v_ids = train_target.iloc[t_idx]['customer_id'], train_target.iloc[v_idx]['customer_id']
    
    X_tr = make_features(train_cust[train_cust['customer_id'].isin(t_ids)], train_fin[train_fin['customer_id'].isin(t_ids)], train_tran[train_tran['customer_id'].isin(t_ids)], REF_DATE_TRAIN)
    X_vl = make_features(train_cust[train_cust['customer_id'].isin(v_ids)], train_fin[train_fin['customer_id'].isin(v_ids)], train_tran[train_tran['customer_id'].isin(v_ids)], REF_DATE_TRAIN)
    X_ts_fold = X_test_base.copy()

    cat_cols = X_tr.select_dtypes(include=['object']).columns.tolist()
    if 'customer_id' in cat_cols: cat_cols.remove('customer_id')
    
    for col in cat_cols:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        X_vl[col] = X_vl[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        X_ts_fold[col] = X_ts_fold[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    X_tr_f, X_vl_f = X_tr.drop(columns=['customer_id']), X_vl.drop(columns=['customer_id'])
    X_ts_f = X_ts_fold.drop(columns=['customer_id'])

    # Churn 학습
    m_c = lgb.LGBMClassifier(**CHURN_PARAMS)
    m_c.fit(X_tr_f, train_target.iloc[t_idx]['target_churn'], eval_set=[(X_vl_f, train_target.iloc[v_idx]['target_churn'])], callbacks=[lgb.early_stopping(300), lgb.log_evaluation(-1)])
    fold_auc = roc_auc_score(train_target.iloc[v_idx]['target_churn'], m_c.predict_proba(X_vl_f)[:, 1])
    auc_scores.append(fold_auc)
    churn_preds += m_c.predict_proba(X_ts_f)[:, 1] / 5
    
    # LTV 학습
    m_l = lgb.LGBMRegressor(**LTV_PARAMS)
    m_l.fit(X_tr_f, np.sqrt(train_target.iloc[t_idx]['target_ltv']), eval_set=[(X_vl_f, np.sqrt(train_target.iloc[v_idx]['target_ltv']))], callbacks=[lgb.early_stopping(200), lgb.log_evaluation(-1)])
    fold_ltv_pred = np.maximum(m_l.predict(X_vl_f), 0) ** 2
    fold_rmse = np.sqrt(mean_squared_error(train_target.iloc[v_idx]['target_ltv'], fold_ltv_pred))
    rmse_scores.append(fold_rmse)
    ltv_preds += (np.maximum(m_l.predict(X_ts_f), 0) ** 2) / 5
    
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

#submission = pd.DataFrame({'customer_id': test_cust['customer_id'], 'target_churn': churn_preds, 'target_ltv': ltv_preds})
#submission.to_csv('submission_final_fixed_report.csv', index=False)
#print("\n✅ submission_final_fixed_report.csv 저장 완료!")