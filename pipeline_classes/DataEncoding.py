class DataTransformer:
    def __init__(self, df, train_df, fields_to_be_encoded):
        self.df = df
        self.train_df = train_df
        self.fields_to_be_encoded = fields_to_be_encoded
        
    def OneHotEncoding(self,keep_original = False, sparse_output=False):
        '''
        ONEHOTENCODING IS BETTER FOR TREE BASED MODELS. 
        HOWEVER A IMPACT ENCODING IS BETTER IF THERE IS A BIG CHANCE OF INCREASE OF VARIABLES 
        NEW UNIQUE VALUES. 
        '''
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        onehot_encoder = OneHotEncoder(sparse_output=sparse_output)  # drop='first' to avoid dummy variable trap

        
        ### FIT TRANSFORMER, SHOULD BE DONE ON WHOLE SET AND NOT ONLY TRAIN SETS LIKE SCALING
        enc = onehot_encoder.fit_transform(self.df[self.fields_to_be_encoded])
        
        ### GET THE NAMES OUT OF THE NEW COLUMNS
        cols = onehot_encoder.get_feature_names_out(self.fields_to_be_encoded)
        
        ### CREATE NEW DF WITH NEW ENCODED COLUMNS
        encoded_df = pd.DataFrame(enc, columns=cols, index=self.df.index)
        
        ### CONCAT AND CREATE A NEW DF WITH NEW.
        ### THIS DOES NOT HOWEVER DROP THE COLUMNS SINCE THEY MIGHT BE NEEDED
        ### ELSEWHERE IN SOME TYPE OF ENCODING.
        df = pd.concat([self.df, encoded_df], axis=1)

        if not keep_original:
            df = df.drop(columns=self.fields_to_be_encoded)

        return df
    
    def TargetEncoding(self, df, target_field, smoothing="auto", cv=5, shuffled=True):
        '''
        Applies target encoding using sklearn's TargetEncoder with cross-validation and smoothing.
        Encodes specified fields in both training and full dataframes using the target field.

        Parameters:
            df (pd.DataFrame): Full dataset to be encoded
            train_df (pd.DataFrame): Training data subset
            fields_to_be_encoded (list): Column(s) to target-encode
            target_field (str): Target column name
            smoothing (str/float): "auto" or float value (default: "auto")
            cv (int): Number of cross-validation folds (default: 5)
            shuffled (bool): Whether to shuffle data (default: True)

        Returns:
            pd.DataFrame: Encoded full dataframe
            pd.DataFrame: Encoded training dataframe
        '''
        from sklearn.preprocessing import TargetEncoder 
        # Initialize encoder with all parameters
        encoder = TargetEncoder(
            smooth=smoothing,
            cv=cv,
            shuffle=shuffled  # Correct parameter name
        )
        
        # Fit and transform on training data
        train_encoded = encoder.fit_transform(
            self.train_df[self.fields_to_be_encoded],  # Correct variable name
            self.train_df[target_field]           # Correct variable name
        )
        
        # Transform full dataset
        df = encoder.transform(
            self.df[self.fields_to_be_encoded]  # Correct variable name
        )
        
        return df, encoder
    
    def ImpactEncoding(self,
                       impact_field, 
                       keep_original = False
                      ):
        '''
        THIS IS NEEDED HERE BECAUSE WE DONT WANT MODEL TO GET A FIXED NUMBER OF COLUMNS BASED ON PRODUCT NUMBER.
        THIS MEANS THAT EVERYTIME DATA GET MORE PRODUCTS IT NEEDS TO BE TRAINED AND HYPERTUNED. 
        INSTEAD WE DO IMPACT ENCODING SO THAT WE CAN USE NEW PRODUCTS INSTEAD AND ALSO SOMEWHAT CLASSIFY THEM INSTEAD.

        THIS CLASS FUNCTION ONLY ACCEPTS 1 COLUMN NOT A LIST OF COLUMNS. 
        '''
        df = self.df.copy()
        if keep_original:
            prefix = 'imp_enc_'
        else:
            prefix =''

        for field in self.fields_to_be_encoded:
            # GROUPBY FIELD AND GET MEAN. 
            mean_encoded = self.train_df.groupby(field)[impact_field].mean().to_dict()
            # MAP THIS TO FIELD.
            df[f'{prefix}{field}'] = df[field].map(mean_encoded)
            
        return df
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

        