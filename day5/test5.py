import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# [기존 경로 설정]
BASE = '/Users/rim/Desktop/workspace/project_1/KDISS-2026/2회 경진대회 데이터'
SEED = 42

# [1] 데이터 로드 (Optuna에서 사용하기 위해 루프 밖에서 로드)
train_cust = pd.read_csv(f'{BASE}/train/train_customer_info.csv')
train_tran = pd.read_csv(f'{BASE}/train/train_transaction_history.csv')
train_fin  = pd.read_csv(f'{BASE}/train/train_finance_profile.csv')
train_target = pd.read_csv(f'{BASE}/train/train_targets.csv')

# [기존 피처 생성 함수 유지]
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
    l1m_data = df_tran[df_tran['trans_date'] >= l1m_limit]
    trend_l1m = l1m_data.groupby('customer_id').agg(amt_l1m=('trans_amount', 'sum'), cnt_l1m=('trans_id', 'count')).reset_index()
    agg = agg.merge(trend_l1m, on='customer_id', how='left').fillna(0)
    agg['amt_trend_ratio'] = (agg['amt_l1m'] - agg['mean_amount']) / (agg['mean_amount'] + 1)
    agg['cnt_trend_ratio'] = (agg['cnt_l1m'] - (agg['frequency']/6)) / ((agg['frequency']/6) + 1)
    df = customer_df.merge(finance_df, on='customer_id', how='left').merge(agg, on='customer_id', how='left')
    df['join_date'] = pd.to_datetime(df['join_date'])
    df['join_days'] = (ref_date - df['join_date']).dt.days
    df['net_cash_flow'] = df['total_deposit_balance'] - df['card_loan_amt']
    df['debt_pressure'] = df['total_loan_balance'] / (df['credit_score'] + 1)
    income_col = 'annual_income' if 'annual_income' in df.columns else 'income_group'
    df['income_val'] = pd.to_numeric(df[income_col], errors='coerce').fillna(0)
    df['loan_to_income'] = df['total_loan_balance'] / (df['income_val'] + 1)
    df['avg_trans_per_month'] = df['frequency'] / ((df['join_days'] / 30) + 1)
    drop_cols = ['gender', 'is_married', 'high_amt_user', 'fin_overdue_days', 'join_date', 'income_val']
    actual_drops = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=actual_drops).fillna(0)

REF_DATE_TRAIN = pd.to_datetime(train_tran['trans_date']).max()

# ==========================================
# 🎯 Optuna 목적 함수 설정
# ==========================================
def churn_objective(trial):
    # 하이퍼파라미터 탐색 범위 정의
    params = {
        'n_estimators': 2000, # 탐색 속도를 위해 적절히 조정
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # Optuna 탐색 중에도 데이터 누수 방지를 위한 3-Fold 교차 검증
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_scores = []
    
    for t_idx, v_idx in cv.split(train_target, train_target['target_churn']):
        t_ids, v_ids = train_target.iloc[t_idx]['customer_id'], train_target.iloc[v_idx]['customer_id']
        
        X_tr = make_features(train_cust[train_cust['customer_id'].isin(t_ids)], train_fin[train_fin['customer_id'].isin(t_ids)], train_tran[train_tran['customer_id'].isin(t_ids)], REF_DATE_TRAIN)
        X_vl = make_features(train_cust[train_cust['customer_id'].isin(v_ids)], train_fin[train_fin['customer_id'].isin(v_ids)], train_tran[train_tran['customer_id'].isin(v_ids)], REF_DATE_TRAIN)
        
        # LabelEncoding (Optuna용 간소화)
        for col in X_tr.select_dtypes(include=['object']).columns:
            if col == 'customer_id': continue
            le = LabelEncoder()
            X_tr[col] = le.fit_transform(X_tr[col].astype(str))
            X_vl[col] = X_vl[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            
        y_tr = train_target.iloc[t_idx]['target_churn']
        y_vl = train_target.iloc[v_idx]['target_churn']
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr.drop(columns=['customer_id']), y_tr)
        
        score = roc_auc_score(y_vl, model.predict_proba(X_vl.drop(columns=['customer_id']))[:, 1])
        cv_scores.append(score)
        
    return np.mean(cv_scores)

# [2] Optuna 실행
print("\n--- ⚡ Optuna 하이퍼파라미터 최적화 시작 ---")
study = optuna.create_study(direction='maximize')
study.optimize(churn_objective, n_trials=20) # 20회
print("\n✅ 최적 파라미터:", study.best_params)
best_churn_params = study.best_params
best_churn_params.update({'n_estimators': 10000, 'verbose': -1}) 