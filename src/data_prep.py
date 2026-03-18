import pandas as pd
from tsfeatures import stl_features, tsfeatures
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datasetsforecast.m4 import M4 

def prepare_data():
    cfg = {
        'data_dir': 'data',
        'm4_group': 'Monthly',
        'seasonality': 12,
        'n_clusters': 3,
        'random_seed': 42,
        'horizon': 12
    }
    
    df, _, _ = M4.load(cfg['data_dir'], cfg['m4_group'])
    df = df[df['unique_id'].isin(df['unique_id'].unique()[:1000])].copy()
    
    feats = tsfeatures(df, freq=cfg['seasonality'], features=[stl_features])
    feats = feats.dropna(axis=1, how='all').dropna()
    
    cols = [f for f in ['trend', 'seasonal_strength', 'linearity', 'spike'] if f in feats.columns]
    
    x_scaled = StandardScaler().fit_transform(feats[cols])
    
    km = KMeans(n_clusters=cfg['n_clusters'], random_state=cfg['random_seed'], n_init='auto')
    feats['cluster'] = km.fit_predict(x_scaled)
    
    df = df[df['unique_id'].isin(feats['unique_id'])].copy()
    df = df.merge(feats[['unique_id', 'cluster']], on='unique_id', how='left')
    df['ds'] = df.groupby('unique_id').cumcount() + 1
    
    test = df.groupby('unique_id').tail(cfg['horizon'])
    train = df.drop(test.index)
    
    return train, test, feats, cfg
