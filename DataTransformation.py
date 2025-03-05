
import pandas as pd
from sklearn.model_selection import train_test_split
class DataTransformer:
    '''
    ALL OF THESE NEED TO BE FITTED ON TRAIN DATA AND NOT THE WHOLE DATA SET. 
    '''
    @staticmethod
    def OneHotEncoding(df, fields_to_be_encoded, keep_original = False, sparse_output=False):
        '''
        ONEHOTENCODING IS BETTER FOR TREE BASED MODELS. 
        HOWEVER A IMPACT ENCODING IS BETTER IF THERE IS A BIG CHANCE OF INCREASE OF VARIABLES 
        NEW UNIQUE VALUES. 
        '''
        from sklearn.preprocessing import OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse_output=sparse_output)  # drop='first' to avoid dummy variable trap

        
        ### FIT TRANSFORMER, SHOULD BE DONE ON WHOLE SET AND NOT ONLY TRAIN SETS LIKE SCALING
        enc = onehot_encoder.fit_transform(df[fields_to_be_encoded])
        
        ### GET THE NAMES OUT OF THE NEW COLUMNS
        cols = onehot_encoder.get_feature_names_out(fields_to_be_encoded)
        
        ### CREATE NEW DF WITH NEW ENCODED COLUMNS
        encoded_df = pd.DataFrame(enc, columns=cols, index=df.index)
        
        ### CONCAT AND CREATE A NEW DF WITH NEW.
        ### THIS DOES NOT HOWEVER DROP THE COLUMNS SINCE THEY MIGHT BE NEEDED
        ### ELSEWHERE IN SOME TYPE OF ENCODING.
        df = pd.concat([df, encoded_df], axis=1)

        if not keep_original:
            df = df.drop(columns=fields_to_be_encoded)

        return df
    @staticmethod
    def TargetEncoding(df, 
                       train_df, 
                       fields_to_be_encoded, 
                       target_field, 
                       smoothing = "auto", 
                       cv = 5,
                       shuffled = True):
        '''
        SAME THING AS IMPACT ENCODING, BUT WITH SKLEARNS OWN LIBRARY WHICH CONTAINS CVS AND SMOOTHING.
        THESE ARE THINGS NOT YET IMPLEMENTED IN OWN MADE IMPACTENCODING.
        
        #target_enc = TargetEncoder(smooth="auto")  # Smoothing value is numeric, not "auto"
        #train_df[ImpactEncoding] = target_enc.fit_transform(train_df[ImpactEncoding], train_df[target])
        #test_X[ImpactEncoding] = target_enc.transform(test_X[ImpactEncoding])
        #enc_df[ImpactEncoding] = target_enc.transform(enc_df[ImpactEncoding])
        '''
        from sklearn.preprocessing import TargetEncoder
        target_enc = TargetEncoder(smooth="auto")  # Smoothing value is numeric, not "auto"
        train_df[targetenc] = target_enc.fit_transform(train_df[targetenc], train_df[target])
        df[targetenc] = target_enc.transform(df[targetenc])
            
        return df
    @staticmethod
    def ImpactEncoding(df, 
                       fields_to_be_encoded, 
                       impact_field, 
                       train_df,
                       keep_original = False
                      ):
        '''
        THIS IS NEEDED HERE BECAUSE WE DONT WANT MODEL TO GET A FIXED NUMBER OF COLUMNS BASED ON PRODUCT NUMBER.
        THIS MEANS THAT EVERYTIME DATA GET MORE PRODUCTS IT NEEDS TO BE TRAINED AND HYPERTUNED. 
        INSTEAD WE DO IMPACT ENCODING SO THAT WE CAN USE NEW PRODUCTS INSTEAD AND ALSO SOMEWHAT CLASSIFY THEM INSTEAD.

        THIS CLASS FUNCTION ONLY ACCEPTS 1 COLUMN NOT A LIST OF COLUMNS. 
        '''

        if keep_original:
            prefix = 'imp_enc_'
        else:
            prefix =''

        for field in fields_to_be_encoded:
            # GROUPBY FIELD AND GET MEAN. 
            mean_encoded = train_df.groupby(field)[impact_field].mean().to_dict()
            # MAP THIS TO FIELD.
            df[f'{prefix}{field}'] = df[field].map(mean_encoded)
            
        return df
        
    @staticmethod
    def MinMaxScaling(df, train_df, 
                      fields_to_be_scaled,
                      feature_range=(0,1)):
        '''
        ### SCALING DOES NOT AFFECT ALL ML ALGOS. ###
        ALGORITHMS THAT FIT A LINE BASED ON WEIGHTED SUM OF INPUT ARE AFFECTED, THIS MEANS THAT ALL
        LINEAR TYPE MODELS LIKE LINEAR REGRESSION, LOGISTIC REGRESSION AND NEURAL NETWORKS, ASWELL AS
        DISTANCE BASED MODELS SUCH AS KNN ARE AFFECTED. TREE TYPE MODELS ARE NOT. 
    
        SHOULD I NORMALIZE OR STANDARDIZE?
        IF THE DISTRIBUTION OF THE QUANTITY IS NORMAL, THEN IT SHOULD BE STANDARDIZE 
        ELSE
        THE DATA SHOULD BE NORMALIZED.
        '''
        
        '''
        USE MINMAX SCALING RANGE OVER 0 WHEN APPLYING POWERTRANSFORMATIONS
        '''
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range = feature_range)
        
        ### FIT_TRANSFORM ON TRAINING SET ONLY TO AVOID LEAKS
        train_df[fields_to_be_scaled] = scaler.fit_transform(train_df[fields_to_be_scaled])

        df[fields_to_be_scaled] = scaler.transform(df[fields_to_be_scaled])
        return df[fields_to_be_scaled], scaler
        
    @staticmethod
    def StandardScaling(df, train_df, 
                        fields_to_be_scaled 
                        ):
        from sklearn.preprocessing import StandardScaler
        # Initialize the StandardScaler
        scaler = StandardScaler()
        
        # Fit the scaler on the training data and transform
        train_df[fields_to_be_scaled] = scaler.fit_transform(train_df[fields_to_be_scaled])

        df[fields_to_be_scaled] = scaler.transform(df[fields_to_be_scaled])
        return df[fields_to_be_scaled], scaler
        
    @staticmethod
    def PowerTransformation(df, train_df, 
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
        
        ### DONT ACTUALLY KNOW IF THIS IS CONSIDERED A DATA LEAK, SO THIS STEP MIGHT BE SOME WHAT UNNECESSARY.
        train_df = pt.fit_transform(train_df)
        df = pt.transform(df)
        return df, pt

    @staticmethod
    def QuantileDistributionTransformer(df, train_df,
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
        train_df = qt.fit_transform(train_df)
        df = qt.transform(df)
        return df, qt
        
    @staticmethod
    def NumericConverted(df, train_df, fields_to_be_converted, bins = 10, encoding = 'onehot', strat = 'uniform'):
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
        kbins = KBinsDiscretizer(n_bins=bins, encode=encoding, strategy=strat)

        ### DONT ACTUALLY KNOW IF THIS IS CONSIDERED A DATA LEAK, SO THIS STEP MIGHT BE SOME WHAT UNNECESSARY.
        train_df[fields_to_be_converted] = kbins.fit_transform(train_df[fields_to_be_converted])
        df[fields_to_be_converted] = kbins.transform(df[fields_to_be_converted])
        return train_df, kbins

    @staticmethod
    def train_test_dataframes(df,x_cols, y_cols, test_size = 0.2, random_state = 42):
        train_X, test_X, train_y, test_y = train_test_split(df[x_cols], df[y_cols], test_size=test_size, random_state=random_state)
        #X
        train_df = train_X.copy()
        train_df[y_cols] = train_y
        
        #y
        test_df = test_X.copy()
        test_df[y_cols] = test_y
        return train_df, test_df
        