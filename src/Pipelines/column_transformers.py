from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
import pandas as pd
import numpy as np


class TotalMonthsEMP(BaseEstimator, TransformerMixin):
    def __init__(self):
        # you could also add parameters here if needed, e.g. capping_percentile
        self.cap_value_ = None

    def fit(self, X, y=None):
        # ensure DataFrame for clarity
        X_ = pd.DataFrame(X).copy()
        # compute cap threshold during fitting
        total_months = X_['months_employed'] + X_['years_employed'] * 12
        q1, q3 = total_months.quantile([0.25, 0.75])
        p90 = total_months.quantile(0.90)
        self.cap_value_ = p90 + 1.5 * (q3 - q1)
        X = X_
        return self  # must return self

    def transform(self, X):
        import pandas as pd
        X_ = pd.DataFrame(X).copy()

        # make sure to handle missing values safely
        X_['total_months_employed'] = (
            X_['months_employed'].fillna(0).infer_objects(copy=False) + X_['years_employed'].fillna(0).infer_objects(copy=False) * 12
        )
        # cap extreme values
        X_.loc[
            X_['total_months_employed'] >= self.cap_value_, 'total_months_employed'
        ] = self.cap_value_
        X['total_months_employed'] = X_['total_months_employed']
        # X.columns = [col.split('__')[-1] for col in X.columns]
        X.drop(columns={'months_employed','years_employed'},inplace=True)
        return X

class CreditScoreOutlier(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        q1, q3 = X['credit_score'].quantile([0.25,0.75])
        p25 = X['credit_score'].quantile(0.25)
        self.lower_cap = p25 - 1.5*(q3-q1)
        return self
    
    def transform(self,X):
        X.loc[X['credit_score'] <= self.lower_cap,'credit_score'] = self.lower_cap
        return X 

##TO be used in Function Transformer
def binary_encoding(X:pd.DataFrame,cols:list):
    for col in cols:
        X[col] = X[col].astype(str).str.lower().map({'yes':1,'no':0})
    return X

def round_off(X:pd.DataFrame,cols:list):
    for col in cols:
        X[col] = np.round(X[col])
    return X

def log_transformation(X,cols:list):
    for col in cols:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X

def rename_column(X:pd.DataFrame):
    X.columns = [col.split('__')[-1] for col in X.columns]
    X = X.loc[:, ~X.columns.duplicated(keep='last')]
    return X

def selected_features(X,features:list):
    X = X[features]
    return X