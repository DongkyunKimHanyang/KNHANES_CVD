import sys
import numpy as np
import pandas as pd
from scipy import interp

import sklearn.metrics as metrics

v=pd.read_csv('./data/variables.csv')

def calc_roc(ytest_one_hot,y_prob):
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    number_of_class=y_prob.shape[1]
    for i in range(number_of_class):
        fpr[i], tpr[i],threshold[i]= metrics.roc_curve(ytest_one_hot[:, i], y_prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"],_= metrics.roc_curve(ytest_one_hot.ravel(), y_prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(number_of_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(number_of_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= number_of_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, threshold, roc_auc



def test_bootstraping(ytest_one_hot,y_prob,ytest,y_predict,out,rng_seed=45,n_bootstraps=1000):
    boot_auroc=[]
    boot_spec=[]
    boot_recall = []
    boot_gmean =[]
    rng=np.random.RandomState(rng_seed)
    number_of_class=y_prob.shape[1]

    for i in range(n_bootstraps):
        indicies = rng.randint(0, len(ytest_one_hot), len(ytest_one_hot))
        if len(np.unique(ytest[indicies])) < number_of_class:
            continue
        if len(np.unique(y_predict[indicies])) < number_of_class:
            continue
        fpr, tpr, threshold, roc_auc = calc_roc(ytest_one_hot[indicies, :], y_prob[indicies, :])
        optimal_idx = np.argmax(np.sqrt(tpr[1] * (1-fpr[1])))
        optimal_threshold = threshold[1][optimal_idx]
        y_predict = (y_prob[:,1]>optimal_threshold).astype(int)
        boot_auroc.append(metrics.roc_auc_score(ytest[indicies], y_prob[indicies,1]))
        recall_temp = metrics.recall_score(ytest[indicies], y_predict[indicies])
        spec_temp = metrics.recall_score(1 - ytest[indicies], 1 - y_predict[indicies])
        boot_spec.append(spec_temp)
        boot_recall.append(recall_temp)
        boot_gmean.append(np.sqrt(recall_temp*spec_temp))


    sorted_gmean = np.sort(np.array(boot_gmean))
    sorted_spec = np.sort(np.array(boot_spec))
    sorted_recall = np.sort(np.array(boot_recall))
    sorted_auroc = np.sort(np.array(boot_auroc))

    print(f"With optimal threshold {optimal_threshold}", file=out)
    fpr, tpr, threshold, roc_auc = calc_roc(ytest_one_hot, y_prob)
    confidence_lower = sorted_gmean[int(0.025 * len(sorted_gmean))]
    confidence_upper = sorted_gmean[int(0.975 * len(sorted_gmean))]
    print("gmean: {:.3f} [{:.3f} - {:.3}]".format(np.mean(sorted_gmean), confidence_lower, confidence_upper), file=out)
    confidence_lower = sorted_spec[int(0.025 * len(sorted_spec))]
    confidence_upper = sorted_spec[int(0.975 * len(sorted_spec))]
    print("specificity: {:.3f} [{:.3f} - {:.3}]".format(np.mean(sorted_spec), confidence_lower, confidence_upper),file=out)
    confidence_lower = sorted_recall[int(0.025 * len(sorted_recall))]
    confidence_upper = sorted_recall[int(0.975 * len(sorted_recall))]
    print("recall: {:.3f} [{:.3f} - {:.3}]".format(np.mean(sorted_recall), confidence_lower, confidence_upper),file=out)
    confidence_lower = sorted_auroc[int(0.025 * len(sorted_auroc))]
    confidence_upper = sorted_auroc[int(0.975 * len(sorted_auroc))]
    print("auroc: {:.3f} [{:.3f} - {:.3}]".format(np.mean(sorted_auroc), confidence_lower, confidence_upper),file=out)



def shap_importance(best_estimator,columns,X_train,X_test,dir,train,plot_=False,model_name='model',label = 'Cardio'):
    import shap
    import joblib
    import os.path
    from joblib import Parallel, delayed
    from joblib import parallel_backend
    from ray.util.joblib import register_ray
    import ray
    from tqdm import tqdm

    print("Start calculating SHAP")
    f = lambda x: best_estimator.predict_proba(x)
    med = pd.DataFrame(X_test).sample(n=10,
                                      random_state=42)  # np.mean(X_test,axis=0).reshape((1, X_test.shape[1]))
    explainer = shap.KernelExplainer(f, med)
    from sklearn.impute import SimpleImputer
    from imblearn.pipeline import Pipeline
    pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy="median")),
    ])
    pipe.fit(X_train)
    X_test = pipe.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=columns)

    #Load or Cacluate the shap values
    if os.path.isfile(dir + 'shap_value.joblib') and not train:
        shap_values = joblib.load(dir + 'shap_value.joblib')
        print('Shap values are loaded')
    else:
        if ray.is_initialized()==False:
            register_ray()
            ray.init(address="auto")
        with parallel_backend('ray'):
            shap_values = Parallel(n_jobs=-1)(delayed(explainer.shap_values)(X_test.iloc[i, :]) for i in tqdm(X_test.index))
            shap_values = np.array(shap_values)[:,1,:]
            joblib.dump(shap_values, dir + 'shap_value.joblib')
    SHAP_ranking = pd.Series(np.mean(np.abs(shap_values),axis=0),index=columns).sort_values(ascending=False)
    SHAP_ranking.to_csv(dir+'shap_ranking.csv',header=False)

    from scipy.stats import spearmanr, pointbiserialr
    SHAP_corr = list()
    for i in range(len(columns)):
        if columns[i] in v.numerical.dropna():
            b = spearmanr(shap_values[:, i], X_test.iloc[:, i]).correlation
        else:
            b = pointbiserialr(shap_values[:, i], X_test.iloc[:, i]).correlation
        SHAP_corr.append(b)
    SHAP_corr = pd.Series(SHAP_corr, index=columns)
    SHAP_corr=SHAP_corr[SHAP_ranking.index]
    SHAP_corr.to_csv(dir+'shap_corr.csv',header=False)
    print("Complete SHAP calculating")
