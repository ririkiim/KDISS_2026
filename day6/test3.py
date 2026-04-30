import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 경로 및 설정
# ==========================================
BASE = "/Users/rim/Desktop/workspace/project_1/KDISS-2026/2회 경진대회 데이터"
SEED = 312

# ==========================================
# 2. 전처리 함수 (기존 로직 유지)
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
        amt_max=('trans_amount', 'max'),
        trans_active_days=('trans_date', lambda x: x.dt.date.nunique())
    ).reset_index()

    agg['amt_cv'] = agg['amt_std'] / (agg['amt_mean'] + 1)
    agg['amount_per_trans'] = agg['amt_sum'] / (agg['frequency'] + 1)
    
    cat_nunique = tran.groupby('customer_id')['item_category'].nunique().rename('cat_nunique')
    
    t_l1m = tran[tran['trans_date'] >= pd.Timestamp('2023-12-01')]
    l1m_sum = t_l1m.groupby('customer_id')['trans_amount'].sum().rename('amt_l1m')
    
    feats = agg.merge(l1m_sum, on='customer_id', how='left').merge(cat_nunique, on='customer_id', how='left')
    
    if target_df is not None:
        df = target_df.merge(cust, on='customer_id', how='left')
    else:
        df = cust.copy()

    df = df.merge(fin, on='customer_id', how='left').merge(feats, on='customer_id', how='left')
    df['join_days'] = (ref_date - pd.to_datetime(df['join_date'])).dt.days
    df['debt_pressure'] = df['total_loan_balance'] / (df['credit_score'] + 1)

    return df.drop(columns=['join_date']).fillna(0)

# 데이터 로드
train_target = pd.read_csv(f'{BASE}/train/train_targets.csv')
train_df_raw = get_features(f'{BASE}/train/train_customer_info.csv', f'{BASE}/train/train_transaction_history.csv', f'{BASE}/train/train_finance_profile.csv', train_target)
test_df_raw = get_features(f'{BASE}/test/test_customer_info.csv', f'{BASE}/test/test_transaction_history.csv', f'{BASE}/test/test_finance_profile.csv')

cat_cols = ['gender', 'region_code', 'prefer_category', 'income_group']

# ==========================================
# 3. 최적 파라미터 (Optuna - Trial 7)
# ==========================================
CHURN_PARAMS = {
    'n_estimators': 10000,
    'learning_rate': 0.002083529297385262,
    'num_leaves': 20,
    'min_child_samples': 44,
    'scale_pos_weight': 4.368290556734767,
    'subsample': 0.6420026989606165,
    'colsample_bytree': 0.8641524423565804,
    'reg_alpha': 0.0075728895828650325,
    'reg_lambda': 1.7298286763168753,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1
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
    'random_state': SEED,
    'verbose': -1
}

# ==========================================
# 4. 최종 학습 및 앙상블 (5-Fold)
# ==========================================
X_train_cat = train_df_raw.drop(columns=['customer_id', 'target_churn', 'target_ltv']).copy()
X_test_cat = test_df_raw[X_train_cat.columns].copy()

train_df = train_df_raw.copy()
test_df = test_df_raw.copy()

# Label Encoding
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    known = set(le.classes_)
    test_df[col] = test_df[col].astype(str).map(lambda x, _le=le, _known=known: int(_le.transform([x])[0]) if x in _known else -1)

X_lgb = train_df.drop(columns=['customer_id', 'target_churn', 'target_ltv'])
X_test_lgb = test_df[X_lgb.columns]
y_churn = train_df['target_churn']
y_ltv_sqrt = np.sqrt(train_df['target_ltv'])

train_df['ltv_bin'] = pd.qcut(train_df['target_ltv'], 5, labels=False, duplicates='drop')
train_df['stratify_col'] = train_df['target_churn'].astype(str) + "_" + train_df['ltv_bin'].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, rmse_scores = [], []
churn_preds, ltv_preds = np.zeros(len(X_test_lgb)), np.zeros(len(X_test_lgb))

print("\n--- 앙상블 학습 시작 ---")
for fold, (t_idx, v_idx) in enumerate(skf.split(X_lgb, train_df['stratify_col'])):
    X_tr_l, X_vl_l = X_lgb.iloc[t_idx], X_lgb.iloc[v_idx]
    X_tr_c, X_vl_c = X_train_cat.iloc[t_idx], X_train_cat.iloc[v_idx]
    yc_tr, yc_vl = y_churn.iloc[t_idx], y_churn.iloc[v_idx]
    yl_tr, yl_vl = y_ltv_sqrt.iloc[t_idx], y_ltv_sqrt.iloc[v_idx]

    # 모델 1: LightGBM (Optuna Best 파라미터 적용)
    m_c_l = lgb.LGBMClassifier(**CHURN_PARAMS).fit(X_tr_l, yc_tr, eval_set=[(X_vl_l, yc_vl)], 
                                                  callbacks=[lgb.early_stopping(300), lgb.log_evaluation(-1)])
    m_l_l = lgb.LGBMRegressor(**LTV_PARAMS).fit(X_tr_l, yl_tr, eval_set=[(X_vl_l, yl_vl)], 
                                                 callbacks=[lgb.early_stopping(200), lgb.log_evaluation(-1)])
    
    # 모델 2: CatBoost (범주형 데이터 강점 활용)
    m_c_cb = CatBoostClassifier(iterations=5000, learning_rate=0.01, depth=6, random_seed=SEED, 
                                verbose=0, cat_features=cat_cols).fit(X_tr_c, yc_tr, eval_set=(X_vl_c, yc_vl))
    m_l_cb = CatBoostRegressor(iterations=3000, learning_rate=0.01, depth=8, random_seed=SEED, 
                               verbose=0, cat_features=cat_cols).fit(X_tr_c, yl_tr, eval_set=(X_vl_c, yl_vl))

    # 앙상블 (LGBM + CatBoost 산술 평균)
    v_churn = (m_c_l.predict_proba(X_vl_l)[:, 1] + m_c_cb.predict_proba(X_vl_c)[:, 1]) / 2
    v_ltv = (np.maximum(m_l_l.predict(X_vl_l), 0)**2 + np.maximum(m_l_cb.predict(X_vl_c), 0)**2) / 2
    
    auc_scores.append(roc_auc_score(yc_vl, v_churn))
    rmse_scores.append(np.sqrt(mean_squared_error(train_df_raw.iloc[v_idx]['target_ltv'], v_ltv)))
    
    # 테스트 데이터 예측 누적
    churn_preds += (m_c_l.predict_proba(X_test_lgb)[:, 1] + m_c_cb.predict_proba(X_test_cat)[:, 1]) / 10
    ltv_preds += (np.maximum(m_l_l.predict(X_test_lgb), 0)**2 + np.maximum(m_l_cb.predict(X_test_cat), 0)**2) / 10
    print(f"Fold {fold+1} 완료: AUC = {auc_scores[-1]:.4f}")

# 결과 출력
mean_auc = np.mean(auc_scores)
mean_rmse = np.mean(rmse_scores)
final_score = 0.5 * mean_auc + 0.5 * (1 / (1 + np.log(mean_rmse + 1)))
print(f"\n=============================================")
print(f"최종 CV AUC 평균: {np.mean(auc_scores):.4f}")
print(f"최종 CV RMSE 평균: {np.mean(rmse_scores):,.0f}")
print(f"대회 산식 최종 Score: {final_score:.5f}")
print(f"=============================================")

# 제출 파일 생성
#submission = pd.DataFrame({
#    'customer_id': test_df_raw['customer_id'],
#    'target_churn': churn_preds,
#    'target_ltv': ltv_preds
#})
#submission.to_csv('submission.csv', index=False)
#print("✅ 제출 파일(submission.csv)이 생성되었습니다.")