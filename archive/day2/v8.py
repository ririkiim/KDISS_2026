"""
==============================================================
v8
--------------------------------------------------------------
1. [LTV] Two-Stage 구조 도입 
   - Stage1: Churn=0 고객만으로 LTV 예측 🟢
   - Stage2: 고LTV(상위 15%) 전용 별도 모델 → 블렌딩 🔴
2. [LTV] 상관분석 결과 반영 → card_loan_amt 중심 피처 강화 🟡
3. [Churn] Gap 0.04~0.06 → 규제 균형점 적용 🟢
==============================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
warnings.filterwarnings('ignore')



EXP = {
    'version'     : 'v8',
    'churn_leaves': 31,
    'churn_alpha' : 2.0,
    'churn_lambda': 10.0,
    'ltv_leaves'  : 63,
    'note'        : 'Two-Stage LTV + card_loan 피처 강화 + Churn Gap 균형점'
}
print(f"\n{'='*60}")
print(f"  실험 버전: {EXP['version']}  |  {EXP['note']}")
print(f"{'='*60}\n")

# ──────────────────────────────────────────────
# 데이터 경로
# ──────────────────────────────────────────────
BASE = '/Users/rim/Desktop/workspace/project_1/KDISS-2026/2회 경진대회 데이터'

tr_info = pd.read_csv(f'{BASE}/train/train_customer_info.csv').copy()
te_info = pd.read_csv(f'{BASE}/test/test_customer_info.csv').copy()
tr_fin  = pd.read_csv(f'{BASE}/train/train_finance_profile.csv').copy()
te_fin  = pd.read_csv(f'{BASE}/test/test_finance_profile.csv').copy()
tr_p    = pd.read_csv(f'{BASE}/train_p.csv').copy()
te_p    = pd.read_csv(f'{BASE}/test_p.csv').copy()

# ──────────────────────────────────────────────
# 전처리
# ──────────────────────────────────────────────
def preprocess_info(df):
    df = df.copy()
    if df['income_group'].dtype == 'object':
        df['income_group'] = df['income_group'].str.extract(r'(\d+)')[0]
    df['income_group'] = pd.to_numeric(df['income_group'], errors='coerce')
    df['income_group'] = df['income_group'].fillna(df['income_group'].median())
    df['join_date'] = pd.to_datetime(df['join_date'])
    df['tenure_days'] = (df['join_date'].max() - df['join_date']).dt.days
    return df

tr_info = preprocess_info(tr_info)
te_info = preprocess_info(te_info)

info_cols = ['customer_id', 'tenure_days', 'income_group', 'age',
             'is_married', 'gender', 'region_code', 'prefer_category', 'join_date']
fin_cols  = ['customer_id', 'fin_asset_trend_score', 'credit_score',
             'total_deposit_balance', 'total_loan_balance',
             'card_loan_amt', 'fin_overdue_days', 'card_cash_service_amt', 'num_active_cards']

overlap    = list(set(info_cols + fin_cols))
tr_p_clean = tr_p.drop(columns=[c for c in overlap if c in tr_p.columns and c != 'customer_id'])
te_p_clean = te_p.drop(columns=[c for c in overlap if c in te_p.columns and c != 'customer_id'])

train = (tr_p_clean
         .merge(tr_fin[fin_cols], on='customer_id', how='left')
         .merge(tr_info[info_cols], on='customer_id', how='left'))
test  = (te_p_clean
         .merge(te_fin[fin_cols], on='customer_id', how='left')
         .merge(te_info[info_cols], on='customer_id', how='left'))

# ──────────────────────────────────────────────
# 피처 엔지니어링
# ──────────────────────────────────────────────
M_COLS = ['amt_m7','amt_m8','amt_m9','amt_m10','amt_m11','amt_m12']

def base_feature_engineering(df):
    df = df.copy()
    existing_m = [c for c in M_COLS if c in df.columns]

    if existing_m:
        df['spending_mean']   = df[existing_m].mean(axis=1)
        df['spending_std']    = df[existing_m].std(axis=1)
        df['spending_min']    = df[existing_m].min(axis=1)
        df['spending_max']    = df[existing_m].max(axis=1)
        df['spending_cv']     = df['spending_std'] / (df['spending_mean'] + 1)
        df['spend_stability'] = df['spending_min'] / (df['spending_mean'] + 1)

        if len(existing_m) >= 6:
            df['recent_3m']      = df[existing_m[-3:]].mean(axis=1)
            df['prev_3m']        = df[existing_m[:3]].mean(axis=1)
            df['spending_trend'] = df['recent_3m'] - df['prev_3m']
            df['spending_accel'] = df['spending_trend'] / (df['prev_3m'] + 1)
            vals = df[existing_m].values
            X_t   = np.arange(len(existing_m), dtype=float)
            X_c   = X_t - X_t.mean()
            df['spending_slope']    = (vals @ X_c) / (X_c @ X_c)
            df['trend_to_noise']    = df['spending_slope'] / (df['spending_std'] + 1)
            df['consec_decline']    = (np.diff(vals, axis=1) < 0).sum(axis=1)
            df['consec_incline']    = (np.diff(vals, axis=1) > 0).sum(axis=1)


    df['net_asset']          = df['total_deposit_balance'] - df['total_loan_balance']
    df['asset_momentum']     = df['total_deposit_balance'] * (df['fin_asset_trend_score'] + 4)
    df['ltv_potential']      = df['credit_score'] * np.log1p(df['total_deposit_balance'].clip(0))
    df['value_per_day']      = df['amt_sum'] / (df['tenure_days'] + 1)
    df['log_deposit']        = np.log1p(df['total_deposit_balance'].clip(0))
    df['log_card_loan']      = np.log1p(df['card_loan_amt'].clip(0))
    df['leverage_ratio']     = df['total_loan_balance'] / (df['total_deposit_balance'] + 1)
    df['cash_advance_ratio'] = df['card_cash_service_amt'] / (df['card_loan_amt'] + 1)
    df['overdue_risk']       = df['fin_overdue_days'] * df['card_cash_service_amt']
    df['clv_proxy']          = (df['spending_mean'] * df['tenure_days']
                                * (1 - df['leverage_ratio'].clip(0, 1)))

    df['card_loan_sq']        = np.sqrt(df['card_loan_amt'].clip(0))          # 분포 정규화
    df['card_loan_per_day']   = df['card_loan_amt'] / (df['tenure_days'] + 1) # 일 평균 카드론
    df['card_loan_to_deposit']= df['card_loan_amt'] / (df['total_deposit_balance'] + 1)  # 예금 대비 카드론 비율
    df['card_loan_to_credit'] = df['card_loan_amt'] / (df['credit_score'] + 1)           # 신용 대비 카드론
    df['total_debt']          = df['card_loan_amt'] + df['total_loan_balance']            # 총 부채
    df['debt_to_deposit']     = df['total_debt'] / (df['total_deposit_balance'] + 1)

    df['trend_x_deposit']     = df['fin_asset_trend_score'] * df['log_deposit']
    df['trend_x_credit']      = df['fin_asset_trend_score'] * df['credit_score']
    df['trend_positive']      = (df['fin_asset_trend_score'] > 0).astype(int)

    df['financial_health']    = (df['credit_score'] * df['fin_asset_trend_score'].clip(0)
                                 / (df['overdue_risk'] + 1))
    df['high_value_signal']   = (df['card_loan_amt'] > 0).astype(int) * df['log_deposit']
    df['spending_x_loan']     = df['spending_mean'] * np.log1p(df['card_loan_amt'].clip(0))

    for col in ['gender', 'region_code', 'prefer_category']:
        df[col] = df[col].astype('category').cat.codes

    df.fillna(0, inplace=True)
    return df

train = base_feature_engineering(train)
test  = base_feature_engineering(test)

# ──────────────────────────────────────────────
# Fold-safe 피처 함수
# ──────────────────────────────────────────────
GROUP_STAT_COLS = ['income_group', 'region_code', 'prefer_category']
TARGET_ENC_COLS = ['region_code', 'prefer_category', 'income_group']

def add_fold_features(tr_fold, val_fold, te_df, target_col, te_accumulator):
    tr_out  = tr_fold.copy()
    val_out = val_fold.copy()
    te_out  = te_df.copy()

    # 그룹 통계 (tr_fold 기준)
    for grp in GROUP_STAT_COLS:
        if grp not in tr_fold.columns:
            continue
        stats       = tr_fold.groupby(grp)['total_deposit_balance'].agg(['mean', 'std'])
        global_mean = tr_fold['total_deposit_balance'].mean()
        global_std  = tr_fold['total_deposit_balance'].std() + 1e-9

        for df_out, df_src in [(val_out, val_fold), (te_out, te_df)]:
            m = df_src[grp].map(stats['mean']).fillna(global_mean)
            s = df_src[grp].map(stats['std']).fillna(global_std)
            df_out[f'deposit_z_in_{grp}'] = (df_src['total_deposit_balance'] - m) / (s + 1e-9)

        tr_grp_mean = tr_fold.groupby(grp)['total_deposit_balance'].transform('mean')
        tr_grp_std  = tr_fold.groupby(grp)['total_deposit_balance'].transform('std').fillna(global_std)
        tr_out[f'deposit_z_in_{grp}'] = (tr_fold['total_deposit_balance'] - tr_grp_mean) / (tr_grp_std + 1e-9)

        for df_out, df_src in [(val_out, val_fold), (te_out, te_df)]:
            rank_map      = tr_fold.groupby(grp)['credit_score'].rank(pct=True)
            grp_rank_mean = tr_fold.assign(_r=rank_map).groupby(grp)['_r'].mean()
            df_out[f'credit_rank_in_{grp}'] = df_src[grp].map(grp_rank_mean).fillna(0.5)
        tr_out[f'credit_rank_in_{grp}'] = tr_fold.groupby(grp)['credit_score'].rank(pct=True)

        # [NEW] card_loan 그룹 통계도 추가
        loan_stats   = tr_fold.groupby(grp)['card_loan_amt'].agg(['mean', 'std'])
        global_lmean = tr_fold['card_loan_amt'].mean()
        global_lstd  = tr_fold['card_loan_amt'].std() + 1e-9
        for df_out, df_src in [(val_out, val_fold), (te_out, te_df)]:
            lm = df_src[grp].map(loan_stats['mean']).fillna(global_lmean)
            ls = df_src[grp].map(loan_stats['std']).fillna(global_lstd)
            df_out[f'loan_z_in_{grp}'] = (df_src['card_loan_amt'] - lm) / (ls + 1e-9)
        tr_lm = tr_fold.groupby(grp)['card_loan_amt'].transform('mean')
        tr_ls = tr_fold.groupby(grp)['card_loan_amt'].transform('std').fillna(global_lstd)
        tr_out[f'loan_z_in_{grp}'] = (tr_fold['card_loan_amt'] - tr_lm) / (tr_ls + 1e-9)

    global_te_mean = tr_fold[target_col].mean()
    for col in TARGET_ENC_COLS:
        if col not in tr_fold.columns:
            continue
        feat_name = f'{col}_te_{target_col}'
        mp = tr_fold.groupby(col)[target_col].mean()

        tr_out[feat_name]  = tr_fold[col].map(mp).fillna(global_te_mean)
        val_out[feat_name] = val_fold[col].map(mp).fillna(global_te_mean)
        te_accumulator.setdefault(feat_name, []).append(
            te_df[col].map(mp).fillna(global_te_mean).values
        )

    return tr_out, val_out


def apply_te_to_test(te_df, te_accumulator, tr_fold, grp_stat_cols):
    te_tmp = te_df.copy()
    for feat_name, arr_list in te_accumulator.items():
        te_tmp[feat_name] = arr_list[-1]
    for grp in grp_stat_cols:
        if grp not in tr_fold.columns:
            continue
        stats       = tr_fold.groupby(grp)['total_deposit_balance'].agg(['mean', 'std'])
        global_mean = tr_fold['total_deposit_balance'].mean()
        global_std  = tr_fold['total_deposit_balance'].std() + 1e-9
        m_ = te_tmp[grp].map(stats['mean']).fillna(global_mean)
        s_ = te_tmp[grp].map(stats['std']).fillna(global_std)
        te_tmp[f'deposit_z_in_{grp}'] = (te_tmp['total_deposit_balance'] - m_) / (s_ + 1e-9)
        rank_map      = tr_fold.groupby(grp)['credit_score'].rank(pct=True)
        grp_rank_mean = tr_fold.assign(_r=rank_map).groupby(grp)['_r'].mean()
        te_tmp[f'credit_rank_in_{grp}'] = te_tmp[grp].map(grp_rank_mean).fillna(0.5)
        # card_loan 그룹 통계
        loan_stats   = tr_fold.groupby(grp)['card_loan_amt'].agg(['mean', 'std'])
        global_lmean = tr_fold['card_loan_amt'].mean()
        global_lstd  = tr_fold['card_loan_amt'].std() + 1e-9
        lm = te_tmp[grp].map(loan_stats['mean']).fillna(global_lmean)
        ls = te_tmp[grp].map(loan_stats['std']).fillna(global_lstd)
        te_tmp[f'loan_z_in_{grp}'] = (te_tmp['card_loan_amt'] - lm) / (ls + 1e-9)
    return te_tmp

# ──────────────────────────────────────────────
# Churn 모델 
# 5-Fold Stratified + 과적합 진단)
# ──────────────────────────────────────────────
DROP_BASE = ['customer_id', 'target_churn', 'target_ltv', 'join_date', 'strat_key']

# Gap 균형점: num_leaves=31, 규제 중간값
churn_params = dict(
    n_estimators     = 3000,
    learning_rate    = 0.008,
    num_leaves       = 24,   # 31
    min_child_samples= 150,
    reg_alpha        = EXP['churn_alpha'],    # 2.0
    reg_lambda       = EXP['churn_lambda'],   # 10.0
    subsample        = 0.7,
    colsample_bytree = 0.7,
    random_state     = 42,
    verbosity        = -1,
)

skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_churn    = np.zeros(len(train))
test_churn   = np.zeros(len(test))
te_acc_churn = {}

print(f"{'='*65}")
print(f"{' [ Churn 모델 학습 + 과적합 진단 ] ':-^65}")
print(f"{'Fold':^6}|{'Train AUC':^11}|{'Val AUC':^11}|{'Gap':^9}|{'Best Iter':^10}")
print(f"{'-'*65}")

for fold, (tr_, val_) in enumerate(skf.split(train, train['target_churn'])):
    tr_fold  = train.iloc[tr_].copy()
    val_fold = train.iloc[val_].copy()

    tr_feat, val_feat = add_fold_features(
        tr_fold, val_fold, test, 'target_churn', te_acc_churn
    )
    ch_feats = [f for f in tr_feat.columns if f not in DROP_BASE]

    m = lgb.LGBMClassifier(**churn_params)
    m.fit(
        tr_feat[ch_feats], tr_fold['target_churn'],
        eval_set=[(val_feat[ch_feats], val_fold['target_churn'])],
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    tr_preds            = m.predict_proba(tr_feat[ch_feats])[:, 1]
    val_preds           = m.predict_proba(val_feat[ch_feats])[:, 1]
    oof_churn[val_]     = val_preds
    tr_auc              = roc_auc_score(tr_fold['target_churn'], tr_preds)
    val_auc             = roc_auc_score(val_fold['target_churn'], val_preds)

    te_tmp = apply_te_to_test(test, te_acc_churn, tr_fold, GROUP_STAT_COLS)
    test_churn += m.predict_proba(
        te_tmp.reindex(columns=ch_feats).fillna(0)
    )[:, 1] / 5

    print(f"Fold {fold+1:>2} |  {tr_auc:.4f}   |  {val_auc:.4f}   | {tr_auc-val_auc:.4f}  | {m.best_iteration_:^10}")

cv_auc = roc_auc_score(train['target_churn'], oof_churn)
print(f"{'-'*65}")
print(f"  ★ OOF AUC: {cv_auc:.4f}")

# ──────────────────────────────────────────────
# LTV 모델 
# Two-Stage 구조
# ──────────────────────────────────────────────
"""
Two-Stage:
  Stage-A: 전체 데이터 → 기본 LTV 예측
  Stage-B: Churn=0 & 고LTV(상위 15%) 고객 → 전용 고가치 모델
  최종: Stage-A 예측 + Stage-B 보정 블렌딩
"""

print(f"\n{'='*65}")
print(f"{' [ LTV 모델 학습 — Two-Stage ] ':-^65}")

y_ltv      = train['target_ltv'].copy()
train_mean = y_ltv.mean()

train['pred_churn'] = oof_churn
test['pred_churn']  = test_churn

y_ltv_pos     = y_ltv.clip(lower=1)
y_bc, lambda_ = boxcox(y_ltv_pos)
print(f"  Box-Cox lambda: {lambda_:.4f}")

HIGH_Q          = 0.85
high_ltv_thresh = y_ltv[train['target_churn'] == 0].quantile(HIGH_Q)
is_high_value   = ((train['target_churn'] == 0) &
                   (y_ltv >= high_ltv_thresh)).values
print(f"  고가치 고객 수: {is_high_value.sum():,}명 "
      f"/ 전체 {len(train):,}명 ({is_high_value.mean():.1%})")

kf           = KFold(n_splits=5, shuffle=True, random_state=42)
oof_ltv_a    = np.zeros(len(train))   # Stage-A (전체)
oof_ltv_b    = np.zeros(len(train))   # Stage-B (고가치 전용)
oof_ltv_b_w  = np.zeros(len(train))   # Stage-B 가중치 합산용
test_ltv_a   = np.zeros(len(test))
test_ltv_b   = np.zeros(len(test))
te_acc_ltv_a = {}
te_acc_ltv_b = {}

LTV_DROP = DROP_BASE + ['pred_churn']

ltv_params_a = dict(
    n_estimators     = 3000,
    learning_rate    = 0.005,
    num_leaves       = EXP['ltv_leaves'],
    min_child_samples= 50,
    reg_alpha        = 0.5,
    reg_lambda       = 2.0,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    verbosity        = -1,
)

# Stage-B: 고가치 고객 전용
ltv_params_b = dict(
    n_estimators     = 300,
    learning_rate    = 0.01,
    num_leaves       = 127,   
    min_child_samples= 20,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    verbosity        = -1,
)


print(f"\n{'─'*65}")
print(f"  [Stage-A] 전체 고객 LTV 모델")
print(f"{'─'*65}")

for fold, (tr_, val_) in enumerate(kf.split(train)):
    tr_fold  = train.iloc[tr_].copy()
    val_fold = train.iloc[val_].copy()

    tr_feat, val_feat = add_fold_features(
        tr_fold, val_fold, test, 'target_ltv', te_acc_ltv_a
    )
    ltv_feats = [f for f in tr_feat.columns if f not in LTV_DROP]

    m_a = lgb.LGBMRegressor(**ltv_params_a)
    m_a.fit(
        tr_feat[ltv_feats], y_bc[tr_],
        eval_set=[(val_feat[ltv_feats], y_bc[val_])],
        callbacks=[lgb.early_stopping(200, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    pred_bc_val      = m_a.predict(val_feat[ltv_feats]).clip(min=y_bc.min())
    oof_ltv_a[val_]  = inv_boxcox(pred_bc_val, lambda_)

    te_tmp = apply_te_to_test(test, te_acc_ltv_a, tr_fold, GROUP_STAT_COLS)
    pred_bc_te = m_a.predict(
        te_tmp.reindex(columns=ltv_feats).fillna(0)
    ).clip(min=y_bc.min())
    test_ltv_a += inv_boxcox(pred_bc_te, lambda_) / 5

    fold_rmse = np.sqrt(mean_squared_error(y_ltv.iloc[val_], oof_ltv_a[val_]))
    print(f"  Fold {fold+1} | RMSE: {fold_rmse:>12,.0f} | best_iter: {m_a.best_iteration_}")

print(f"\n{'─'*65}")
print(f"  [Stage-B] 고가치 고객(Churn=0, 상위 15%) 전용 모델")
print(f"{'─'*65}")

# Stage-B
for fold, (tr_, val_) in enumerate(kf.split(train)):
    tr_fold  = train.iloc[tr_].copy()
    val_fold = train.iloc[val_].copy()

    tr_high_mask  = is_high_value[tr_]
    val_high_mask = is_high_value[val_]

    if tr_high_mask.sum() < 50:   
        print(f"  Fold {fold+1} | 고가치 고객 부족 ({tr_high_mask.sum()}명) → Skip")
        continue

    tr_feat_all, val_feat_all = add_fold_features(
        tr_fold, val_fold, test, 'target_ltv', te_acc_ltv_b
    )
    ltv_feats_b = [f for f in tr_feat_all.columns if f not in LTV_DROP]

    tr_feat_h  = tr_feat_all[tr_high_mask]
    val_feat_h = val_feat_all[val_high_mask]
    y_tr_bc_h  = y_bc[tr_][tr_high_mask]
    y_val_h    = y_ltv.iloc[val_].values[val_high_mask]

    m_b = lgb.LGBMRegressor(**ltv_params_b)
    m_b.fit(
        tr_feat_h[ltv_feats_b], y_tr_bc_h)

    pred_bc_val_h = m_b.predict(val_feat_h[ltv_feats_b]).clip(min=y_bc.min())
    oof_ltv_b[val_][val_high_mask]   += inv_boxcox(pred_bc_val_h, lambda_)
    oof_ltv_b_w[val_][val_high_mask] += 1

    te_tmp = apply_te_to_test(test, te_acc_ltv_b, tr_fold, GROUP_STAT_COLS)
    pred_bc_te_b = m_b.predict(
        te_tmp.reindex(columns=ltv_feats_b).fillna(0)
    ).clip(min=y_bc.min())
    test_ltv_b += inv_boxcox(pred_bc_te_b, lambda_) / 5

    fold_rmse_h = np.sqrt(mean_squared_error(y_val_h, inv_boxcox(pred_bc_val_h, lambda_)))
    print(f"  Fold {fold+1} | 고가치 RMSE: {fold_rmse_h:>12,.0f} | "
          f"고가치 N: {tr_high_mask.sum():,} | best_iter: {m_b.best_iteration_}")

oof_ltv_b = np.where(oof_ltv_b_w > 0, oof_ltv_b / oof_ltv_b_w, oof_ltv_a)

# ── 최종 블렌딩 ──
# 고가치 고객: Stage-A 40% + Stage-B 60%
# 일반 고객  : Stage-A 100%
blend_weight = np.where(is_high_value, 0.6, 0.0)
oof_ltv_final  = (1 - blend_weight) * oof_ltv_a + blend_weight * oof_ltv_b

test_high_mask_prob = test_churn < np.quantile(test_churn, 0.15) 
te_blend = np.where(test_high_mask_prob, 0.6, 0.0)
test_ltv_final = (1 - te_blend) * test_ltv_a + te_blend * test_ltv_b

# ── 분위수 보정 ──
def quantile_calibration(oof_pred, y_true, test_pred, n_bins=10):
    bins        = np.percentile(oof_pred, np.linspace(0, 100, n_bins + 1))
    bins[0]    -= 1e-9
    bins[-1]   += 1e-9
    cal         = test_pred.copy()
    bin_oof     = np.digitize(oof_pred, bins) - 1
    bin_te      = np.digitize(test_pred, bins) - 1
    for b in range(n_bins):
        m_oof = bin_oof == b
        if m_oof.sum() == 0:
            continue
        scale        = y_true[m_oof].mean() / (oof_pred[m_oof].mean() + 1e-9)
        cal[bin_te == b] = test_pred[bin_te == b] * scale
    return cal

test_ltv_cal = quantile_calibration(oof_ltv_final, y_ltv.values, test_ltv_final)

# OOF RMSE (Mean Matching 보정)
oof_scale      = train_mean / oof_ltv_final.mean()
cv_rmse_scaled = np.sqrt(mean_squared_error(y_ltv, oof_ltv_final * oof_scale))

# ── LTV 분포 진단 ──
print(f"\n  [ LTV 분포 진단 ]")
print(f"  {'분위수':^6} | {'OOF':^12} | {'Train 실제':^12} | {'Test 예측':^12}")
print(f"  {'-'*52}")
for q in [0.25, 0.50, 0.75, 0.90, 0.95]:
    print(f"  Q{q:.0%}  | {np.quantile(oof_ltv_final,q):>12,.0f} | "
          f"{np.quantile(y_ltv,q):>12,.0f} | "
          f"{np.quantile(test_ltv_cal,q):>12,.0f}")

# ──────────────────────────────────────────────
# 최종 점수 및 출력
# ──────────────────────────────────────────────
ltv_part = 1 / (1 + np.log10(cv_rmse_scaled))
total    = 0.5 * cv_auc + 0.5 * ltv_part

print(f"\n{'='*65}")
print(f"{' [ 최종 통합 Score ] ':-^65}")
print(f"  Churn AUC (OOF) : {cv_auc:.4f}")
print(f"  LTV RMSE        : {cv_rmse_scaled:>12,.0f} 원")
print(f"  LTV 변환 점수    : {ltv_part:.4f}")
print(f"  {'─'*45}")
print(f"  통합 Score       : {total:.4f}  (0.5×AUC + 0.5×LTV)")
print(f"{'='*65}")

# ──────────────────────────────────────────────
# 제출
# ──────────────────────────────────────────────
# submission = pd.DataFrame({
#    'customer_id' : test['customer_id'],
#    'target_churn': np.clip(test_churn, 0, 1),
#    'target_ltv'  : np.clip(test_ltv_cal, 0, None),
# })

# out_path = f"{BASE}/submission_{EXP['version']}.csv"
# submission.to_csv(out_path, index=False, encoding='utf-8-sig')

# print(f"\n  ✅ 저장 완료: {out_path}")
print(f"\n  [ 분포 비교 (Test vs Train) ]")
print(f"  - Churn Mean : {test_churn.mean():.4f}  (Train: {train['target_churn'].mean():.4f})")
print(f"  - LTV Mean   : {test_ltv_cal.mean():>12,.0f} 원  (Train: {train_mean:>12,.0f} 원)")

print(f"\n  [ 실험 설정 요약 — {EXP['version']} ]")
for k, v in EXP.items():
    print(f"    {k}: {v}")