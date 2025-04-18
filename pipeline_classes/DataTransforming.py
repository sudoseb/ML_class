import pandas as pd

class DataTransformers:
    def __init__(self, df, train_df):
        self.df = df
        self.train_df = train_df
        
    def PowerTransformation(self, standardize=True, method='yeo-johnson'):
        '''
        Forces gaussian-like distribution. Use this when data is close to normal,
        and contains some outliers.

        Box-Cox: strictly positive values only.
        Yeo-Johnson: works with zero and negative values.
        '''
        from sklearn.preprocessing import PowerTransformer

        pt = PowerTransformer(method=method, standardize=standardize)
        train_df = pt.fit_transform(self.train_df)
        df = pt.transform(self.df.copy())
        return df, pt

    def QuantileDistributionTransformer(self, n_qt=100, distribution='normal'):
        '''
        Transforms features to follow a normal or uniform distribution.

        Normal: default and common.
        Uniform: useful if outliers are common.

        Note: This may be considered data leakage depending on context.
        '''
        from sklearn.preprocessing import QuantileTransformer

        qt = QuantileTransformer(n_quantiles=n_qt, output_distribution=distribution)
        train_df = qt.fit_transform(self.train_df)
        df = qt.transform(self.df.copy())
        return df, qt

    @staticmethod
    def LogTransformer(column):
        '''
        Applies a log1p (log(1 + x)) transformation. Reduces skew, good for strictly positive values.
        
        Accepts either a Series or DataFrame.
        '''
        import numpy as np
        return np.log1p(column)
