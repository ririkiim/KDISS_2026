import pandas as pd

BASE = '/Users/rim/Desktop/workspace/project_1'

c_df = pd.read_csv(f'{BASE}/pred_churn.csv')
l_df = pd.read_csv(f'{BASE}/pred_ltv.csv')

final = pd.merge(c_df, l_df, on='customer_id')

final.to_csv(f'{BASE}/submission_final.csv', index=False, encoding='utf-8-sig')

print(f"✅ submission_final.csv 생성 완료")