from dataload import Dataset
from utils import test_bootstraping, shap_importance
from VIF import VIF_filter
from Boruta import BorutaShap

import argparse
import numpy as np
import pandas as pd
import joblib
import ray
from joblib import parallel_backend
from ray.util.joblib import register_ray
import os,sys

from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

v=pd.read_csv('./data/variables.csv')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def tuning(xtrain,ytrain,model,param_grid,columns):
    from imblearn.pipeline import Pipeline
    continuous_ind = np.array([True if feature in list(v.numerical.dropna()) else False for feature in columns])
    pipe = Pipeline([
        ('scaler', ColumnTransformer([('somename', StandardScaler(), continuous_ind)], remainder='passthrough')),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy="median")),
        ('sampling_under', RandomUnderSampler(random_state=0, sampling_strategy=0.5)),
        ('sampling_over', SMOTE(random_state=0, sampling_strategy=1)),
        ('model', model)
    ])
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1,
                        return_train_score=True)
    grid.fit(xtrain, ytrain)

    return grid

model_dict ={
    'SVM_model':SVC(probability=True, random_state=42),
    'SVM_param_grid':{'model__C':np.power(10.0, np.linspace(-2, 2, 20)),'model__kernel':['rbf','sigmoid'],'model__gamma':['scale','auto',0.1,1,10]},
    'MLP_model':MLPClassifier(max_iter=6500, early_stopping=True,n_iter_no_change=100, random_state=42),
    'MLP_param_grid':{'model__solver':['adam','sgd'],'model__hidden_layer_sizes' :[(20,20),(40,40),(60,60),(20,20,20),(40,40,40),(60,60,60),(80,80,80),(100,100,100)],
                      'model__activation' : ['logistic','tanh','relu'],'model__alpha': [0.0001,0.0003,0.0006,0.0009]},
    'RandomForest_model':RandomForestClassifier(random_state=42),
    'RandomForest_param_grid':{'model__n_estimators' : [100,200,400,700,1000],'model__max_depth' : [4,6,8,10,12],'model__min_samples_split' : [5,10,30,50,70,100],'model__min_samples_leaf' : [5,10,30,50,70,100]},
    'LightGBM_model': lgb.LGBMClassifier(random_state=42,n_estimators=250),
    'LightGBM_param_grid': {'model__boosting_type':['gbdt','dart'],'model__max_depth' : [8,9],'model__min_child_samples': [10,20,40,60,80,100],
                        'model__num_leaves':[120,170,250,300,350,400],'model__learning_rate': [0.1,0.05],'model__subsample': [0.6,0.8],}
}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="Cardio")
    parser.add_argument("--data_path", type=str, default="./data/sample.csv")
    parser.add_argument("--model", type=str, default="SVM", choices=['SVM','RandomForest','LightGBM','MLP'])
    parser.add_argument("--VIF", action='store_true')
    parser.add_argument("--Boruta", action='store_true')
    parser.add_argument("--train", action='store_true')
    args = parser.parse_args()

    result_diretory = f'./result/{args.model}/{args.label}'
    try:
        if not os.path.exists(result_diretory):
            os.makedirs(result_diretory)
    except OSError:
        print('Error: Creating directory.')

    data = Dataset(args.data_path, args.label)
    xtrain, xtest, ytrain, ytest, columns = data.load_dataset()

    if args.VIF:
        if os.path.isfile('./result/VIF_accepted.joblib'):
            accepted_columns = joblib.load('./result/VIF_accepted.joblib')
            rejected_columns = joblib.load('./result/VIF_rejected.joblib')
        else:
            VIF = VIF_filter(xtrain,columns, 4)
            accepted_columns, rejected_columns =  VIF.filtering()
            joblib.dump(accepted_columns, './result/VIF_accepted.joblib')
            joblib.dump(rejected_columns, './result/VIF_rejected.joblib')
        xtrain = xtrain[accepted_columns]
        xtest = xtest[accepted_columns]

    if args.Boruta:
        if os.path.isfile('./result/Boruta_accepted.joblib'):
            accepted_columns = joblib.load('./result/Boruta_accepted.joblib')
            rejected_columns = joblib.load('./result/Boruta_rejected.joblib')
        else:
            Feature_Selector = BorutaShap(model=RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42),pvalue=0.05)
            accepted_columns, rejected_columns = Feature_Selector.fit(X=xtrain, y=ytrain, use_tentative=True, verbose=True)
            joblib.dump(accepted_columns, './result/Boruta_accepted.joblib')
            joblib.dump(rejected_columns, './result/Boruta_rejected.joblib')
        xtrain = xtrain[accepted_columns]
        xtest = xtest[accepted_columns]



    if args.train:
        register_ray()
        if not ray.is_initialized():
            ray.init(address="auto")
        print(ray.is_initialized())
        with parallel_backend('ray'):
            grid = tuning(xtrain, ytrain, model_dict[args.model+'_model'],model_dict[args.model+'_param_grid'],accepted_columns)
            joblib.dump(grid, result_diretory + f'/grid_{args.model}_{args.label}.joblib')
    else:
        grid = joblib.load(result_diretory + f'/grid_{args.model}_{args.label}.joblib')


    best_params = grid.best_params_
    best_score = grid.best_score_
    best_train_score = grid.cv_results_['mean_train_score'][grid.best_index_]
    best_estimator = grid.best_estimator_
    out = open(result_diretory + f'/Bootstrap_results.txt', 'w')
    print('Best params:', best_params, file=out)
    print('Best train_score:', best_train_score, file=out)
    print('Best score:', best_score, file=out)
    y_predict = best_estimator.predict(xtest)
    y_prob = best_estimator.predict_proba(xtest)
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder()
    enc.fit(ytest.values.reshape(-1, 1))
    ytest_one_hot = enc.transform(ytest.values.reshape(-1, 1)).toarray()
    test_bootstraping(ytest_one_hot, y_prob, ytest.values, y_predict,out, rng_seed=45, n_bootstraps=1000)
    out.close()

    np.random.seed(42)
    selected_xtest = xtest.sample(frac=0.1,random_state=40)
    shap_importance(best_estimator, accepted_columns, xtrain,
                    selected_xtest,
                    dir=result_diretory + '/', train=args.train,
                    plot_=False,
                    model_name=f'{args.model}', label=args.label)
