import warnings 
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection      import train_test_split
from sklearn.feature_selection    import f_regression, mutual_info_regression
class feature_selection:
    '''
    FOR REGRESSION PROBLEMS. IF CLASSIFICATION OTHER SELECTIONS LIKE CHI-X TEST SHOULD BE USED FOR STATISTICAL ANALYSIS.
    '''
    
    @staticmethod
    def categorical_output_selection():
        ### numeric to categorical 
        # Chi2 test
        # Mutual Information Statistics
        #from sklearn.feature_selection    import chi2
        raise NotImplementedError
        
    @staticmethod    
    def numeric_fs(score_function,X_train, y_train, numericals, verbose  = True):
        '''
        THERE IS A WAYS OF DOING THIS, FOR EXAMPLE;
        mutual_info_regression
        f_regression
        '''
        from sklearn.feature_selection    import SelectKBest
        # configure to select all features
        fsm = SelectKBest(score_func=score_function, k='all')
        
        # learn relationship from training data
        fsm.fit(X_train, y_train)
        
        # transform train input dat
        X_train_fsm = fsm.transform(X_train[numericals])
        scores = {numericals[i]: fsm.scores_[i] for i in range(len(fsm.scores_))}       
        metrics = {}

        if verbose:
            for i in range(len(fsm.scores_)):
                metrics[numericals[i]] = fsm.scores_[i]
                print('%s - %d: %f' % (numericals[i], i, fsm.scores_[i]))

        sorted_features = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return scores, sorted_features
        
    @staticmethod   
    def recursive_regression_selection(model, X_train, y_train, scoring='explained_variance', fs_model = DecisionTreeRegressor(), n_splits = 5, n_repeats = 2, random_state = 1, verbose = True):
        '''
        THIS IS ONLY FOR NUMERIC FEATURES, THIS MEANS CATEGORICALS NEED TO BE ENCODED.
        '''
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedKFold
        from sklearn.feature_selection import RFECV
        from sklearn.pipeline import Pipeline
        
        rfe = RFECV(estimator=fs_model)
        pipeline = Pipeline(steps=[('s',rfe), ('m',model)])
        
        # evaluate model
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)

        rfe.fit(X_train, y_train)
        
        # summarize all features
        if verbose:
            for i in range(X_train.shape[1]):
                print('Column: %s, Selected=%s, Rank: %d' % (X_train.columns[i], rfe.support_[i], rfe.ranking_[i]))

        #SAVING 
        columns_to_keep = [X_train.columns[i] for i in range(len(rfe.support_)) if rfe.support_[i]]
        
        return columns_to_keep
        
    @staticmethod
    def permutation_importance():
        return NotImplementedError
    
