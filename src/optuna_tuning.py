import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. 목적 함수 정의
def churn_objective(trial):
    params = {
        'n_estimators': 1000, 
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 100, 'verbose': -1, 'n_jobs': -1
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
    scores = []

    # 데이터 전처리 및 학습 (데이터프레임 train_df_raw가 미리 정의되어 있어야 함)
    for t_idx, v_idx in skf.split(train_df_raw, train_df_raw['target_churn']):
        X_tr, X_vl = train_df_raw.iloc[t_idx].copy(), train_df_raw.iloc[v_idx].copy()
        y_tr, y_vl = X_tr['target_churn'], X_vl['target_churn']
        
        for col in ['gender', 'region_code', 'prefer_category', 'income_group']:
            le = LabelEncoder()
            X_tr[col] = le.fit_transform(X_tr[col].astype(str))
            X_vl[col] = X_vl[col].astype(str).map(lambda x, _le=le: int(_le.transform([x])[0]) if x in _le.classes_ else -1)
            
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr.drop(columns=['customer_id', 'target_churn', 'target_ltv']), y_tr)
        preds = model.predict_proba(X_vl.drop(columns=['customer_id', 'target_churn', 'target_ltv']))[:, 1]
        scores.append(roc_auc_score(y_vl, preds))
        
    return np.mean(scores)

# 2. 최적화 실행 및 결과 출력
study = optuna.create_study(direction='maximize')
study.optimize(churn_objective, n_trials=20)

print("\n" + "="*30)
print("BEST PARAMETERS")
print("="*30)
print(study.best_params)
print("="*30)
print(f"Best AUC: {study.best_value:.4f}")