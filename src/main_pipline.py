import pandas as pd
import os
from catboost import CatBoostRegressor
from mlforecast import MLForecast
from mlforecast.target_transforms import LocalBoxCox, Differences
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape, mase
from functools import partial

from transforms import Log1pTransform
from data_prep import prepare_data

def run_pipeline():
    train, test, feats, cfg = prepare_data()
    
    t_dict = {
        "Target_None": [],
        "Target_Log1p": [Log1pTransform()],
        "Target_BoxCox": [LocalBoxCox()],
        "Target_Diff": [Differences([1])]
    }

    ml_preds = []
    for name, t_list in t_dict.items():
        mlf = MLForecast(
            models={name: CatBoostRegressor(iterations=500, random_seed=42, verbose=False)},
            freq=1,
            lags=[1, 12],
            target_transforms=t_list
        )
        mlf.fit(train[['unique_id', 'ds', 'y']])
        ml_preds.append(mlf.predict(h=cfg['horizon']))

    sf = StatsForecast(
        models=[Naive(), SeasonalNaive(cfg['seasonality']), AutoETS(cfg['seasonality'])],
        freq=1,
        n_jobs=-1
    )
    all_preds = sf.forecast(df=train[['unique_id', 'ds', 'y']], h=cfg['horizon']).reset_index()

    for p in ml_preds:
        all_preds = all_preds.merge(p, on=['unique_id', 'ds'], how='left')

    res_df = test[['unique_id', 'ds', 'y', 'cluster']].merge(all_preds, on=['unique_id', 'ds'])
    
    mase_with_seasonality = partial(mase, seasonality=cfg['seasonality'])
    mase_with_seasonality.__name__ = 'mase'
    model_cols = [c for c in all_preds.columns if c not in ['unique_id', 'ds']]
    
    eval_res = evaluate(
        df=res_df,
        metrics=[smape, mase_with_seasonality],
        train_df=train,
        models=model_cols
    )
    eval_res = eval_res.merge(feats[['unique_id', 'cluster']], on='unique_id')
    report = eval_res.groupby(['metric', 'cluster']).mean(numeric_only=True).round(3)

    os.makedirs('results', exist_ok=True)
    report.to_csv('results/final_report.csv')

if __name__ == "__main__":
    run_pipeline()
