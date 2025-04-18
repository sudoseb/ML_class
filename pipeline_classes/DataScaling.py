import pandas as pd
from sklearn.model_selection import train_test_split

class Scalers:
    '''
    Handles feature scaling with proper train-test separation to prevent data leakage
    
    Key Principles:
    1. Always fit scalers on training data only
    2. Transform both training and test data using trained scaler
    3. Preserve original dataframe structure
    
    When to Use:
    - Essential for distance-based algorithms (KNN, SVM) and gradient-based models (NNs, linear models)
    - Optional for tree-based models (Random Forests, XGBoost)
    - Critical when features have different units/ranges
    
    Example:
    >>> scaler = Scaler(full_df, ['age', 'income'], train_df)
    >>> scaled_data, mm_scaler = scaler.MinMaxScaling()
    '''
    
    def __init__(self, df, fields_to_be_scaled, train_df=False):
        '''
        Parameters:
        df: Full dataset (including test data)
        fields_to_be_scaled: Features requiring scaling
        train_df: Training subset (False if not split yet)
        '''
        self.df = df
        self.fields_to_be_scaled = fields_to_be_scaled
        self.train_df = train_df

    def MinMaxScaling(self, feature_range=(0,1)):
        '''
        Squeezes values into specified range (default 0-1)
        
        When to Use:
        - Non-normal distributions
        - Neural networks (especially with sigmoid/tanh activations)
        - Image data (pixel values)
        - When preserving zero values is important
        
        Rule of Thumb:
        Use before PowerTransforms that require positive inputs
        Prefer over StandardScaler if data contains outliers
        
        Best Practice:
        Set feature_range=(0,1) for vanilla cases
        Use (0.1,0.9) to leave margin for unseen extremes
        '''
        from sklearn.preprocessing import MinMaxScaler
        
        # Initialize scaler with specified range
        scaler = MinMaxScaler(feature_range=feature_range)
        
        # Prevent leakage: Only fit on training data
        self.train_df[self.fields_to_be_scaled] = scaler.fit_transform(
            self.train_df[self.fields_to_be_scaled]
        )

        # Apply to full dataset
        self.df[self.fields_to_be_scaled] = scaler.transform(
            self.df[self.fields_to_be_scaled]
        )
        
        return self.df[self.fields_to_be_scaled], scaler
        
    def StandardScaling(self):
        '''
        Centers data (μ=0) and scales to unit variance (σ=1)
        
        When to Use:
        - Normally distributed features
        - Linear models (Regression, PCA, LDA)
        - SVM with RBF kernel
        - When comparing feature importance magnitudes
        
        Rule of Thumb:
        Check for normality using Q-Q plots first
        Sensitive to outliers - use RobustScaler if >3% outliers
        
        Warning:
        Avoid if data contains significant outliers
        Not suitable for sparse data
        '''
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        self.train_df[self.fields_to_be_scaled] = scaler.fit_transform(
            self.train_df[self.fields_to_be_scaled]
        )

        self.df[self.fields_to_be_scaled] = scaler.transform(
            self.df[self.fields_to_be_scaled]
        )
        
        return self.df[self.fields_to_be_scaled], scaler
    
    def RobustScaling(self):
        '''
        Scales using median and IQR (Outlier-resistant)
        
        When to Use:
        - Data with significant outliers
        - Non-normal distributions
        - High-dimensional datasets
        - When using Median-based statistics
        
        Rule of Thumb:
        Default choice for messy real-world data
        Use with quantile-based models (e.g., Quantile Regression)
        
        Note:
        Preserves outlier structure - doesn't eliminate outliers
        IQR range: 25th-75th percentile
        '''
        from sklearn.preprocessing import RobustScaler
        
        scaler = RobustScaler()
        self.train_df[self.fields_to_be_scaled] = scaler.fit_transform(
            self.train_df[self.fields_to_be_scaled]
        )

        self.df[self.fields_to_be_scaled] = scaler.transform(
            self.df[self.fields_to_be_scaled]
        )
        
        return self.df[self.fields_to_be_scaled], scaler

    def MaxAbsScaling(self):
        '''
        Scales each feature by its maximum absolute value.
        Good for sparse data to preserve zero entries.
        '''
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        
        self.train_df[self.fields_to_be_scaled] = scaler.fit_transform(
            self.train_df[self.fields_to_be_scaled]
        )
        self.df[self.fields_to_be_scaled] = scaler.transform(
            self.df[self.fields_to_be_scaled]
        )
        return self.df[self.fields_to_be_scaled], scaler

    def InverseTransform(self, data, scaler):
        '''
        Restores scaled data to original values.
        
        Parameters:
        - data: Scaled data
        - scaler: The fitted scaler used originally
        '''
        return scaler.inverse_transform(data)

    def ScalerTests(self):
        '''
        Try multiple scalers and return scaled versions for experimentation.
        '''
        results = {}
        for method in [self.MinMaxScaling, self.StandardScaling, self.RobustScaling, self.MaxAbsScaling]:
            try:
                scaled, _ = method()
                results[method.__name__] = scaled
            except Exception as e:
                results[method.__name__] = f"Error: {e}"
        return results
    

    def SaveScaler(self, scaler, path):
        import joblib
        joblib.dump(scaler, path)

    @staticmethod
    def LoadScaler(path):
        import joblib
        return joblib.load(path)
