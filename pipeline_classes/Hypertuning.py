from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from skopt import BayesSearchCV
from skopt.space import Real, Integer

'''
class XGBhypertuning:
    def __init__(self, baysian = True, objective='reg:pseudohubererror', enable_categorical=True, max_depth=8, min_child_weight=1, subsample=1, colsample_bytree=0.1, gamma=0, reg_lambda=1, reg_alpha=0.1, random_state=42]):
        self.baysian            = baysian
        self.objective          = objective
        self.enable_categorical = enable_categorical
        self.max_depth          = max_depth
        self.min_child_weight   = min_child_weight,
        self.subsample          = subsample,
        self.colsample_bytree   = colsample_bytree, 
        self.gamma              = gamma,
        self.reg_lambda         = reg_lambda,
        self.reg_alpha          = reg_alpha,
        self.random_state       = random_state,
    ('learning_rate': Real(0.01, 0.3, prior='log-uniform'),  # Learning rate
    'max_depth': Integer(3, 10),                           # Max depth of the trees
    'n_estimators': Integer(50, 300),                      # Number of trees
    'min_child_weight': Integer(1, 10),                    # Min child weight
    'subsample': Real(0.5, 1.0),                           # Subsample ratio
    'colsample_bytree': Real(0.5, 1.0),                    # Colsample by tree
    'gamma': Real(0, 5),                                   # Gamma parameter
    'reg_lambda': Real(0, 10),                             # L2 regularization
    'reg_alpha': Real(0, 10)                               # L1 regularization
{
    'learning_rate': uniform(0.01, 0.3 - 0.01),  # Range around the static model's learning rate
    'n_estimators': randint(100, 500)
}
# Define the XGBoost model
'''
''' -------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------- '''
'''                               RANDOM HYPERTUNING                           '''
''' -------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------- '''
'''
random_xgb_model = xgb.XGBRegressor(
    objective='reg:pseudohubererror',
    enable_categorical=True,
    max_depth=8,           # Match max_depth from static model
    min_child_weight=1,
    subsample=1,
    colsample_bytree=0.8, # Match from static model
    gamma=0,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=42       # Ensure reproducibility
)

# Define the hyperparameter distribution for Random Search
param_distributions = {
    'learning_rate': uniform(0.01, 0.3 - 0.01),  # Range around the static model's learning rate
    'n_estimators': randint(100, 500)
}

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=60,  # Number of parameter settings to sample
    cv=5,       # Cross-validation
    n_jobs=-1,  
    verbose=1
)

# Fit the model with Bayesian optimization
random_search.fit(train_X, train_y,eval_set=[(test_X, test_y)])



'''




''' -------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------- '''
'''                               BAYSIAN HYPERTUNING                          '''
''' -------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------- '''

'''
# Define your model
xgb_model = xgb.XGBRegressor(
    objective='reg:pseudohubererror',
    enable_categorical=True,
    random_state=42
)

# Define the search space for hyperparameters
search_space = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),  # Learning rate
    'max_depth': Integer(3, 10),                           # Max depth of the trees
    'n_estimators': Integer(50, 300),                      # Number of trees
    'min_child_weight': Integer(1, 10),                    # Min child weight
    'subsample': Real(0.5, 1.0),                           # Subsample ratio
    'colsample_bytree': Real(0.5, 1.0),                    # Colsample by tree
    'gamma': Real(0, 5),                                   # Gamma parameter
    'reg_lambda': Real(0, 10),                             # L2 regularization
    'reg_alpha': Real(0, 10)                               # L1 regularization
}

# Bayesian Optimization using BayesSearchCV
opt = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=search_space,
    n_iter=30,        # Number of iterations
    cv=5,             # Cross-validation folds
    n_jobs=-1,        # Use all available cores
    random_state=42
)
'''