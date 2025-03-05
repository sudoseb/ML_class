
import pandas as pd
from sklearn.model_selection import train_test_split
class DataTransformer:
    def __init__(self, df, train_df):
        self.df = df
        self.train_df
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
        
    def NumericConverted(self, fields_to_be_converted, bins = 10, encoding = 'onehot', strat = 'uniform'):
        '''
         Uniform:  Each bin has the same width in the span of possible values for the variable.
                    A uniform discretization transform will preserve the probability distribution of each input
                    variable but will make it discrete with the specified number of ordinal groups or labels (strat = uniform)
            
         Quantile: Each bin has the same number of values, split based on percentiles.
                    A quantile discretization transform will attempt to split the observations for each input variable
                    into k groups, where the number of observations assigned to each group is approximately equal. (strat = quantile)
            
         Clustered: Clusters are identified and examples are assigned to each group.
                     A k-means discretization transform will attempt to fit k clusters for each input variable and
                     then assign each observation to a cluster. (strat = kmeans)
        '''
        '''
        kbins = KBinsDiscretizer(n_bins=bins, encode=encoding, strategy=strat)

        ### DONT ACTUALLY KNOW IF THIS IS CONSIDERED A DATA LEAK, SO THIS STEP MIGHT BE SOME WHAT UNNECESSARY.
        train_df[fields_to_be_converted] = kbins.fit_transform(train_df[fields_to_be_converted])
        df[fields_to_be_converted] = kbins.transform(df[fields_to_be_converted])
        '''
        return NotImplementedError
