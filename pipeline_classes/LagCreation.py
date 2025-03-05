class Temporal:
    def __init__(self, df, n_lags, target_fields, groupby_field, date_field, ascending):
        self.df = df
        self.n_lags = n_lags
        self.target_fields = target_fields
        self.groupby_field = groupby_field
        self.date_field = date_field
        self.ascending = ascending

    def rolling_lags(self):
        import pandas as pd
        # Sort values by the grouping and date fields
        #self.df = self.df.sort_values(by=[self.groupby_field, self.date_field], ascending = self.ascending)
        
        # Generate lags for each target field
        for target in self.target_fields:                
            # Generate lags for other target fields as well
            for lag in range(1, self.n_lags + 1):
                self.df[f'lag__{lag}_{target}'] = self.df.copy().groupby(self.groupby_field)[target].shift(lag)
                
        return self.df
    def rolling_metric(self, metric):
        import pandas as pd
        metrics = {'sum':True, 'mean':False}
        # Sort values by the grouping and date fields
        #self.df = self.df.sort_values(by=[self.groupby_field, self.date_field], ascending = self.ascending)
        
        result_df = self.df.copy()
        for target in self.target_fields:
            for lag in range(1, self.n_lags + 1):
                # Apply rolling sum or mean based on the selected metric
                if metrics[metric]:
                    result_df[f'rolling__{metric}_{lag}_{target}'] = (
                        self.df.groupby(self.groupby_field)[target]
                        .rolling(lag)
                        .sum()
                        .reset_index(level=0, drop=True)
                    )
                else:
                    result_df[f'rolling__{metric}_{lag}_{target}'] = (
                        self.df.groupby(self.groupby_field)[target]
                        .rolling(lag)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )

        return result_df
        
    def static_rolling_metric(self, metric, label):
        import pandas as pd
        result_df = self.df.copy()
        result_df[f'rolling_{self.n_lags}_{metric}__{label}'] = (
                        self.df.groupby(self.groupby_field)[label]
                        .rolling(self.n_lags)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
        return result_df