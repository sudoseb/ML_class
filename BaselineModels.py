import xgboost as xgb

class Baseline:
    def __init__(self, 
                 objective='reg:pseudohubererror', 
                 learning_rate=0.03, 
                 max_depth=8, 
                 n_estimators=100,
                 min_child_weight=1, 
                 subsample=1,
                 colsample_bytree=1,
                 gamma=0,
                 cv=5,
                 eval_metric='mae',
                 enable_categorical=True, 
                 random_state=42,
                 early_stopping_rounds=100,
                 verbose=0, 
                 eval_set=False
                ):
    
        self.objective = objective
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.eval_metric = eval_metric
        self.enable_categorical = enable_categorical
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.cv = cv
        self.verbose = verbose

    def XGBoost(self):
        xgb_model = xgb.XGBRegressor(
            objective=self.objective,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            eval_metric=self.eval_metric,
            enable_categorical=self.enable_categorical,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            verbosity=self.verbose
        )
        return xgb_model
    