import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import os

# ──────────────────────────────────────────────
BASE = '/Users/rim/Desktop/workspace/project_1/KDISS-2026/2회 경진대회 데이터'

customer = pd.read_csv(f'{BASE}/train_customer_info.csv')
transaction = pd.read_csv(f'{BASE}/train_transaction_history.csv')
finance = pd.read_csv(f'{BASE}/train_finance_profile.csv')
target = pd.read_csv(f'{BASE}/train_targets.csv')

datasets = {
    'customer'    : customer,
    'transaction' : transaction,
    'finance'     : finance,
    'target'      : target
}

for name, df in datasets.items():
    print(f"\n{'='*45}")
    print(f"  [{name}]  {df.shape[0]:,}행 × {df.shape[1]}열")
    print(f"{'='*45}")
    print(df.dtypes)

datasets = {
    'customer'    : customer,
    'transaction' : transaction,
    'finance'     : finance,
    'target'      : target
}

for name, df in datasets.items():
    print(f"\n{'='*45}")
    print(f"  [{name}]  {df.shape[0]:,}행 × {df.shape[1]}열")
    print(f"{'='*45}")
    print(df.dtypes)

datasets = {
    'customer'    : customer,
    'transaction' : transaction,
    'finance'     : finance,
    'target'      : target
}

for name, df in datasets.items():
    print(f"\n{'='*45}")
    print(f"  [{name}]  {df.shape[0]:,}행 × {df.shape[1]}열")
    print(f"{'='*45}")
    print(df.dtypes)



# ── Churn 분포 ──
churn_vc  = target['target_churn'].value_counts().sort_index()
churn_pct = (churn_vc / len(target) * 100).round(2)

print("[ Churn 분포 ]")
print(pd.DataFrame({'count': churn_vc, '%': churn_pct}))
print(f"\n  → 이탈률: {target['target_churn'].mean():.2%}")

# ── LTV 기초통계 & 왜도 ──
ltv = target['target_ltv']

print("\n[ LTV 기초통계 ]")
print(ltv.describe().round(0))

skewness = ltv.skew()
print(f"\n  왜도(skewness): {skewness:.2f}")

if skewness > 1:
    print("  → 오른쪽 치우침 심함 ⚠️  log 변환 필수")
elif skewness > 0.5:
    print("  → 약간 치우침, log 변환 권장")
else:
    print("  → 정규분포에 가까움 ✅")

# ── log 변환 후 왜도 비교 ──
ltv_log = np.log1p(ltv)
print(f"  log 변환 후 왜도: {ltv_log.skew():.2f}")

# 날짜 변환 (아직 안 했을 경우)
transaction['trans_date'] = pd.to_datetime(transaction['trans_date'])

# ── 거래 기간 ──
print("[ 거래 기간 ]")
print(f"  시작: {transaction['trans_date'].min().date()}")
print(f"  종료: {transaction['trans_date'].max().date()}")

# ── 월별 거래 건수 ──
print("\n[ 월별 거래 건수 ]")
transaction['month'] = transaction['trans_date'].dt.to_period('M')
print(transaction.groupby('month').size().to_frame('거래건수'))

# ── 고객 1인당 거래 건수 ──
print("[ 고객 1인당 거래 건수 ]")
tx_per_customer = transaction.groupby('customer_id').size()
print(tx_per_customer.describe().round(1))

# ── biz_type / item_category 분포 ──
print("\n[ biz_type 분포 ]")
print(transaction['biz_type'].value_counts())

print("\n[ item_category 분포 ]")
print(transaction['item_category'].value_counts())

# 연체일수 분포 (churn 신호 핵심 변수)
print("[ fin_overdue_days ]")
print(finance['fin_overdue_days'].describe().round(1))
print(f"  연체 있는 고객 수: {(finance['fin_overdue_days'] > 0).sum():,}명")

# 신용점수 범위 확인
print("\n[ credit_score ]")
print(finance['credit_score'].describe().round(1))

# 기준일 설정 (거래 마지막 날 다음날)
reference_date = pd.Timestamp('2024-01-01')

transaction['trans_date'] = pd.to_datetime(transaction['trans_date'])

# ── Recency / Frequency / Monetary ──
agg_base = transaction.groupby('customer_id').agg(
    recency        = ('trans_date', lambda x: (reference_date - x.max()).days),
    frequency      = ('trans_id', 'count'),
    total_amount   = ('trans_amount', 'sum'),
    mean_amount    = ('trans_amount', 'mean'),
    std_amount     = ('trans_amount', 'std'),
    max_amount     = ('trans_amount', 'max'),
    min_amount     = ('trans_amount', 'min'),
).reset_index()

print(agg_base.shape)
agg_base.head(3)

# ── 최근 1개월 vs 이전 소비 비교 (소비 감소 추세 → churn 신호) ──
last_1m = transaction[transaction['trans_date'] >= pd.Timestamp('2023-12-01')]
prev_5m = transaction[transaction['trans_date'] <  pd.Timestamp('2023-12-01')]

last_1m_agg = last_1m.groupby('customer_id')['trans_amount'].sum().rename('amt_last_1m')
prev_5m_agg = prev_5m.groupby('customer_id')['trans_amount'].sum().rename('amt_prev_5m')

agg_trend = pd.concat([last_1m_agg, prev_5m_agg], axis=1).fillna(0)

# 소비 감소율 (음수 = 감소)
agg_trend['amt_trend_ratio'] = (
    agg_trend['amt_last_1m'] - agg_trend['amt_prev_5m'] / 5
) / (agg_trend['amt_prev_5m'] / 5 + 1)

agg_trend = agg_trend.reset_index()
print(agg_trend.shape)
agg_trend.head(3)

# ── 구매 간격 (gap) ──
agg_gap = transaction.groupby('customer_id')['trans_date'].apply(
    lambda x: x.sort_values().diff().dt.days.mean()
).rename('mean_purchase_gap').reset_index()

print(agg_gap.shape)
agg_gap.head(3)

# ── 채널 비율 (Online 비율) ──
agg_channel = transaction.groupby('customer_id').apply(
    lambda x: (x['biz_type'] == 'Online').mean(), include_groups=False
).rename('online_ratio').reset_index()

print(agg_channel.shape)

# ── 카테고리 다양성 ──
agg_category = transaction.groupby('customer_id').agg(
    category_nunique = ('item_category', 'nunique'),
    top_category     = ('item_category', lambda x: x.value_counts().index[0]),
).reset_index()

print(agg_category.shape)

# ── 할부 비율 ──
agg_install = transaction.groupby('customer_id')['is_installment'].mean()\
              .rename('installment_ratio').reset_index()

print(agg_install.shape)

# transaction 집계 병합
tx_features = agg_base\
    .merge(agg_trend,    on='customer_id', how='left')\
    .merge(agg_gap,      on='customer_id', how='left')\
    .merge(agg_channel,  on='customer_id', how='left')\
    .merge(agg_category, on='customer_id', how='left')\
    .merge(agg_install,  on='customer_id', how='left')

# 전체 병합
df = target\
    .merge(customer,    on='customer_id', how='left')\
    .merge(finance,     on='customer_id', how='left')\
    .merge(tx_features, on='customer_id', how='left')

print(f"최종 shape: {df.shape}")
df.head(3)

# ── join_date → 가입 경과일 ──
df['join_date'] = pd.to_datetime(df['join_date'])
df['join_days'] = (reference_date - df['join_date']).dt.days
df = df.drop(columns=['join_date'])

# ── 범주형 인코딩 ──
cat_cols = ['gender', 'region_code', 'prefer_category', 'income_group', 'top_category']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("인코딩 완료")
print(df[cat_cols].head(3))

# ── LTV log 변환 ──
df['target_ltv_log'] = np.log1p(df['target_ltv'])

print(f"LTV 왜도 원본     : {df['target_ltv'].skew():.2f}")
print(f"LTV 왜도 log 변환 : {df['target_ltv_log'].skew():.2f}")

# ── 최종 feature 확인 ──
print(f"최종 shape : {df.shape}")
print(f"컬럼 목록  :\n{df.columns.tolist()}")

drop_cols = ['customer_id', 'target_churn', 'target_ltv', 'target_ltv_log']

X = df.drop(columns=drop_cols)
y_churn = df['target_churn']
y_ltv = df['target_ltv_log']

print(f"X shape      : {X.shape}")
print(f"y_churn 분포 : {y_churn.value_counts().to_dict()}")
print(f"y_ltv   범위 : {y_ltv.min():.2f} ~ {y_ltv.max():.2f}")

from sklearn.model_selection import train_test_split

X_train, X_val, y_churn_train, y_churn_val, y_ltv_train, y_ltv_val = train_test_split(
    X, y_churn, y_ltv,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y_churn
)

print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")

