import numpy as np
from mlforecast.target_transforms import BaseTargetTransform

class Log1pTransform(BaseTargetTransform):
    def fit_transform(self, df):
        df = df.copy()
        df['y'] = np.log1p(df['y'])
        return df

    def inverse_transform(self, df):
        df = df.copy()
        cols_to_transform = df.columns.drop(['unique_id', 'ds'], errors='ignore')
        for col in cols_to_transform:
            df[col] = np.expm1(df[col])
        return df
