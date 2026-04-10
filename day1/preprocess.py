import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

BASE = '/Users/rim/Desktop/workspace/project_1'

train_cust   = pd.read_csv(f'{BASE}/train/train_customer_info.csv')
train_trans  = pd.read_csv(f'{BASE}/train/train_transaction_history.csv')
train_fin    = pd.read_csv(f'{BASE}/train/train_finance_profile.csv')
train_target = pd.read_csv(f'{BASE}/train/train_targets.csv')
test_cust    = pd.read_csv(f'{BASE}/test/test_customer_info.csv')
test_trans   = pd.read_csv(f'{BASE}/test/test_transaction_history.csv')
test_fin     = pd.read_csv(f'{BASE}/test/test_finance_profile.csv')

def build_features(cust_df, trans_df, fin_df, target_df=None):
    df = pd.merge(cust_df, fin_df, on='customer_id', how='left')
    if target_df is not None: 
        df = pd.merge(df, target_df, on='customer_id', how='left')

    df['join_date'] = pd.to_datetime(df['join_date'])
    trans_df['trans_date'] = pd.to_datetime(trans_df['trans_date'])
    trans_df['month'] = trans_df['trans_date'].dt.month
    ref_date = trans_df['trans_date'].max()
    df['days_since_joined'] = (ref_date - df['join_date']).dt.days

    for col in ['gender', 'region_code', 'prefer_category', 'income_group']:
        le = LabelEncoder(); df[col] = le.fit_transform(df[col].astype(str))

    # 순자산(예금-대출), 부채비율, 카드현금서비스 비율
    df['net_asset'] = df['total_deposit_balance'] - df['total_loan_balance']
    df['debt_to_deposit'] = df['total_loan_balance'] / (df['total_deposit_balance'] + 1)
    df['cash_service_ratio'] = df['card_cash_service_amt'] / (df['total_deposit_balance'] + 1)
    
    # 연체일수, 현금서비스, 카드론, 총대출을 합쳐 위험도 수치화
    dv = ['fin_overdue_days','card_cash_service_amt','card_loan_amt','total_loan_balance']
    dz = df[dv].copy()
    for v in dv: dz[v] = (dz[v] - dz[v].mean()) / (dz[v].std() + 1e-9)
    df['fin_distress_v2'] = dz.sum(axis=1)

    # 총 거래횟수, 총 결제액, 평균 결제액, 최대 결제액, 결제액 표준편차, 최근 결제일 계산
    agg = trans_df.groupby('customer_id').agg(
        trans_count=('trans_id','count'), amt_sum=('trans_amount','sum'), 
        amt_mean=('trans_amount','mean'), amt_max=('trans_amount','max'), 
        amt_std=('trans_amount','std'), recency=('trans_date', lambda x: (ref_date-x.max()).days)
    ).reset_index()
    agg['spending_per_trans'] = agg['amt_sum'] / (agg['trans_count'] + 1)
    df = pd.merge(df, agg, on='customer_id', how='left')

    # 7월부터 12월까지 각 달의 총 결제액과 결제 횟수
    m_amt = trans_df.groupby(['customer_id','month'])['trans_amount'].sum().unstack(fill_value=0)
    m_amt.columns = [f'amt_m{int(c)}' for c in m_amt.columns]
    m_cnt = trans_df.groupby(['customer_id','month'])['trans_id'].count().unstack(fill_value=0)
    m_cnt.columns = [f'cnt_m{int(c)}' for c in m_cnt.columns]
    df = df.merge(m_amt.reset_index(), on='customer_id', how='left')
    df = df.merge(m_cnt.reset_index(), on='customer_id', how='left')
    
    # 전반기(7-9월) 대비 후반기(10-12월) 결제액 성장률
    df['amt_first_half'] = df[['amt_m7','amt_m8','amt_m9']].sum(axis=1)
    df['amt_second_half'] = df[['amt_m10','amt_m11','amt_m12']].sum(axis=1)
    df['half_growth_ratio'] = df['amt_second_half'] / (df['amt_first_half'] + 1)

    # 6개월 중 실제로 결제한 달이 몇 달인지 계산
    df['active_months'] = (df[[c for c in df.columns if c.startswith('amt_m')]] > 0).sum(axis=1)

    # 상대 순위: 소득그룹 내 신용점수 순위, 지역 내 결제액 순위
    df['credit_rank_in_income'] = df.groupby('income_group')['credit_score'].rank(pct=True)
    df['spend_rank_in_region'] = df.groupby('region_code')['amt_sum'].rank(pct=True)

    df.fillna(0, inplace=True)
    return df

train_p = build_features(train_cust, train_trans, train_fin, train_target)
test_p = build_features(test_cust, test_trans, test_fin)

train_p.to_csv(f'{BASE}/train_p.csv', index=False, encoding='utf-8-sig')
test_p.to_csv(f'{BASE}/test_p.csv', index=False, encoding='utf-8-sig')
print("✅ 데이터 저장 완료")