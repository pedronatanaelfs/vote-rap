# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:13:08 2024
"""

import pandas as pd
import json
import os
import datetime as dt
import pickle

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, RepeatedKFold, LeaveOneOut, TimeSeriesSplit, ShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

import warnings
warnings.filterwarnings('ignore')

src_path = os.path.abspath(os.getcwd())
root_path = os.path.dirname(src_path)

data_path = os.path.join(root_path, 'dados')
raw_path = os.path.join(data_path, 'raw')
interim_path = os.path.join(data_path, 'interim')
processed_path = os.path.join(data_path, 'processed')
models_path = os.path.join(root_path, 'models')
config_path = os.path.join(root_path, 'config')

with open(os.path.join(config_path,'model.json'), encoding='latin1') as json_file:
    model_configs = json.load(json_file)  

rand = model_configs['random']
n_jobs = model_configs['jobs']
n_iter = model_configs['iterations']
n_folds = model_configs['folds']
models = model_configs['models']
time_serie_split = model_configs['time_serie_split']
split_vote = model_configs['split_vote']
train_size = model_configs['train_size']
cv_method = model_configs['cv_method']


dict_ml_hyperparams = pickle.load(open(os.path.join(config_path, "hyperparams.sav"), 'rb'))
dict_cv_method = pickle.load(open(os.path.join(config_path, "cv.sav"), 'rb'))


def split_data(df, X, y, train_size, id_cols, target, rand, time_serie_split = True, split_vote = False):
    
    """
    Splits the dataset into training and testing sets. Supports both time series split and random split.
    
    Parameters:
    - df: DataFrame containing the full dataset.
    - X: DataFrame of feature variables (excluding the target).
    - y: Series containing the target variable.
    - train_size: Float value (between 0 and 1) indicating the proportion of data to use for training.
    - id_cols: List of columns to be used as the index for the dataset.
    - target: String representing the target column name.
    - rand: Integer used for random state, ensuring reproducibility when random splitting.
    - time_serie_split: Boolean flag to determine if the split should preserve time order (default: True).
    - split_vote: Boolean flag to use specific logic for splitting based on 'idVotacao' (default: False).
    
    Returns:
    - X_train: DataFrame of training features.
    - X_test: DataFrame of testing features.
    - y_train: Series/DataFrame of training target values.
    - y_test: Series/DataFrame of testing target values.
    """
    
    if time_serie_split:            
        if split_vote:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                shuffle = False, 
                                                                test_size= 1 - train_size, 
                                                                random_state=rand)    
            y_test = y_test.to_frame()

        else: 
            df_size = df.groupby('idVotacao', as_index=False).agg({'data':['size', 'last']})
            df_size.columns = ['_'.join(x).replace('data_','').replace('_','') for x in df_size.columns]
            df_size = df_size.sort_values('last')
            
            df_size['cum_size'] = df_size['size'].cumsum()/len(df)
            
            votes_train = df_size[df_size['cum_size']<train_size]['idVotacao'].to_list()
            votes_test = df_size[df_size['cum_size']>=train_size]['idVotacao'].to_list()
            
            X_train = df[df['idVotacao'].isin(votes_train)].set_index(id_cols).drop(target,axis=1)
            X_test = df[df['idVotacao'].isin(votes_test)].set_index(id_cols).drop(target,axis=1)
            y_train = df[df['idVotacao'].isin(votes_train)].set_index(id_cols)[target]
            y_test =  df[df['idVotacao'].isin(votes_test)].set_index(id_cols)[[target]]        
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            shuffle = True, 
                                                            test_size= 1 - train_size, 
                                                            random_state=rand)
        y_test = y_test.to_frame()    
    
    return X_train, X_test, y_train, y_test 



def run_ml_model(model_name, X_train, y_train, cv_method, fit = True, random_cv = True):
    
    """
    Runs a machine learning model with cross-validation and hyperparameter tuning.
    
    Parameters:
    - model_name: String representing the name of the model to be used ('logit', 'ridge', 'knn', 'svm', 'rf', 'bagging', 'gbm', or 'benchmark').
    - X_train: DataFrame of training features.
    - y_train: Series of training target values.
    - cv_method: String representing the cross-validation method to be used (must correspond to a key in dict_cv_method).
    - fit: Boolean indicating whether the model should be fit to the training data (default: True).
    - random_cv: Boolean to determine whether to use RandomizedSearchCV (True) or GridSearchCV (False) for hyperparameter tuning (default: True).
    
    Returns:
    - The fitted model if `fit=True`, otherwise the hyperparameter tuning object (RandomizedSearchCV or GridSearchCV).
    """
    
    cv = dict_cv_method[cv_method]        
    cv_index = [x for x in cv.split(X_train)]

    params_ = dict_ml_hyperparams[model_name]

    if model_name in ['logit', 'benchmark']:
        model_ = LogisticRegression()
    elif model_name == 'ridge':
        model_ = RidgeClassifier()
    elif model_name == 'knn':
        model_ = KNeighborsClassifier()
    elif model_name == 'svm':
        model_ = SVC()
    elif model_name == 'rf':
        model_ = RandomForestClassifier()
    elif model_name == 'bagging':
        model_ = BaggingClassifier()
    elif model_name == 'gbm':
        model_ = GradientBoostingClassifier()

    if random_cv:
        model_grid = RandomizedSearchCV(model_, 
                                        param_distributions=params_, 
                                        n_iter=n_iter, 
                                        cv=cv_index, 
                                        scoring='accuracy',
                                        n_jobs=n_jobs,
                                        random_state=rand,
                                        verbose=False
                                        )             
    else:
        model_grid = GridSearchCV(model_,
                                  params= params_, 
                                  cv=cv_index, 
                                  scoring='accuracy',
                                  n_jobs=n_jobs,
                                  verbose=False
                                  )
 
    if fit:
        model_fit = model_grid.fit(X_train, y_train) 
        if model_name == 'benchmark':
            return model_.fit(X_train, y_train)
        else:
            return model_fit
    return model_grid



if __name__ == "__main__":
    
    df_meta = pd.read_excel(os.path.join(interim_path, 'metadata.xlsx'), sheet_name = 'metadata')    
    df_votos = pd.read_csv(os.path.join(interim_path, 'features.csv'), sep = ';')
   
    target = df_meta[df_meta['type'] == 'Target']['var'].item()
    id_cols  = df_meta[df_meta['type'] == 'Index']['var'].to_list()    
    features = df_meta[df_meta['type'] == 'Features']['var'].to_list()

    df_votos[target].value_counts()
    df_votos = df_votos[df_votos[target].isin(['Sim','Não'])]
    df_votos = df_votos.sort_values(['data', 'y', 'idProposicao', 'idVotacao', 'idDeputado'])
        
    version = 2 #this version remove de orientations columns
    sufix = '' if version!=2 else f"_v{version}"
    
    # metrics_dict = {}
    y_test_agg = pd.DataFrame()
    legis_list = df_votos['idLegislatura'].drop_duplicates().sort_values().to_list()
    legis_list = [52, 53, 54, 55, 56]  
    fit = False
    for legis in sorted(legis_list):
        
        # metrics_dict[legis] = {}
        df_votos_f = df_votos[df_votos['idLegislatura'] == legis]

        if legis == 'overall':
            df_votos_f = df_votos.copy()

        df_std = df_votos_f.std().reset_index()
        df_std.columns = ['var', 'std']

        df_std = df_std[~ df_std['var'].isin(id_cols + [target])]
        cols_to_drop = df_std[df_std['std'] == 0]['var'].to_list()

        df_votos_f = df_votos_f.drop(cols_to_drop,axis=1)

        cols_ori = [x for x in df_votos_f.columns if 'd_ori' in x]
        # df_votos_f = df_votos_f.drop(cols_ori,axis=1)

        X = df_votos_f.set_index(id_cols).drop(target,axis=1)
        y = df_votos_f.set_index(id_cols)[target]

        X_train, X_test, y_train, y_test = split_data(df_votos_f, X, y, train_size, id_cols, target, rand, time_serie_split, split_vote)

        del X, y, df_votos_f
        estimators_list = []
        for model_name in models:
            
            if fit:
                print(f'running {model_name} and legis {legis}')
                time_beg = dt.datetime.now()            
                if model_name == 'benchmark':                
    
                    cols_ori = [x for x in X_train.columns if 'd_ori' in x]
                    X_train_bench = X_train[cols_ori]
                    X_test_bench = X_test[cols_ori]
    
                    model_fit = run_ml_model(model_name, X_train_bench, y_train, cv_method)
                    pred_values = [x for x in model_fit.predict(X_test_bench)]    
                else:
                    if version == 2:
                        X_train = X_train.drop(cols_ori,axis=1)
                        
                    model_fit = run_ml_model(model_name, X_train, y_train, cv_method)
                    pred_values = [x for x in model_fit.predict(X_test)]
                    
                model_filename = os.path.join(models_path, f"model_{model_name}_{legis}{sufix}.sav")                                
                pickle.dump(model_fit, open(model_filename, 'wb'))
        
                y_test[f'{target}_{model_name}'] =  pred_values

            else:
                print(f'running {model_name} and legis {legis}')

                model_grid = run_ml_model(model_name, X_train, y_train, cv_method, fit = False)
                estimators_list.append((model_name, model_grid))
                y_test  = pd.DataFrame()

        if fit:
            y_test_agg = pd.concat([y_test_agg,y_test])
        else:
            for cl in ['soft', 'hard']:
                try:
                    time_beg = dt.datetime.now()            

                    vot_cl = VotingClassifier(estimators = estimators_list, voting = cl, n_jobs = -1, verbose = False) 
                    vot_cl_fit = vot_cl.fit(X_train, y_train) 

                    time_end = dt.datetime.now()            
                    diff_time = time_end - time_beg
                    print(diff_time)

                    model_filename = os.path.join(models_path, f"model_vc_soft_{cl}_{legis}{sufix}.sav")                                
                    pickle.dump(vot_cl_fit, open(model_filename, 'wb'))

                except:
                    print('not able to run hard')
