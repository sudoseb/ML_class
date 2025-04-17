
import pandas as pd
class DataTransformer:
    def __init__(self, df, train_df):
        self.df = df
        self.train_df = train_df
    def PowerTransformation(self,
                            standardize = True,  
                            method = 'yeo-johnson' ):
        '''
        Forces gaussian like distribution. Use this when almost a gaussian distribution, and when containing some outliers.
        
        Sometimes a lift in performance can be achieved by first standardizing the raw dataset prior
        to performing a Yeo-Johnson transform. We can explore this by adding a StandardScaler as a
        first step in the pipeline.

        As for Box-Cox that assumes the values of the input variable to which it is applied are strictly positive. That
        means 0 and negative values are not supported.
        '''
        
        # can be box-cox or yeo-johnsson
        from sklearn.preprocessing import PowerTransformer
        
        pt = PowerTransformer(method=method, standardize = standardize)
        train_df = pt.fit_transform(self.train_df)
        df = pt.transform(self.df.copy())
        return df, pt

    @staticmethod
    def QuantileDistributionTransformer(self,
                                        n_qt=100,
                                        distribution = 'normal'):
        '''
        Normaldistribution most cases; 
        Else if the data is not data with a large and sparse range of
        values and rarely for regression but e.g. outliers that are common rather than rare - Use uniform. 

        n_qt can be hypertuned, use this outside of class ... probably. 
        '''
        from sklearn.preprocessing import QuantileTransformer

        qt = QuantileTransformer(n_quantiles = n_qt, output_distribution=distribution)
        
        ### DONT ACTUALLY KNOW IF THIS IS CONSIDERED A DATA LEAK, SO THIS STEP MIGHT BE SOME WHAT UNNECESSARY.
        train_df = qt.fit_transform(self.train_df)
        df = qt.transform(self.df.copy())
        return df, qt
    @staticmethod
    def LogTransformer(self, column):
        '''
        Reduces skew; good for strictly positive values.
        Since this is row-wise transformation, it doesnt need to be done to only a training set since no data leak is happening.
        '''
        import numpy as np
        return np.log1p(column)
     
