import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from time import time

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from rule_ensemble import RuleFitClassifier, GeneralizedLinerRuleClassifier, LocallyInterpretableRuleEnsembleClassifier
from datasets import Dataset


METHODS = [
    'RF',
    'LightGBM', 
    'KernelSVM', 
    'RuleFit', 
    'GLRM', 
    'LIRE',
]


def run(dataset='a', params={}, max_depth=3, n_estimators=100, n_splits=10):
    np.random.seed(0)
    D = Dataset(dataset=dataset)
    print('#', D.dataset_fullname, 'dataset')
    print()

    keys = ['method', 'fold', 'accuracy', 'f1', 'auc', 'time', 'gsup', 'lsup', 'rsup', 'div']
    res = dict( [(key, []) for key in keys] )

    k = 0
    for tr, ts in StratifiedKFold(n_splits=n_splits).split(D.X, D.y):
        k = k + 1
        X_tr, X_ts, y_tr, y_ts = D.X[tr], D.X[ts], D.y[tr], D.y[ts]
        forest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators).fit(X_tr, y_tr)
 
        print('## Fold =', k)
        for method in METHODS:

            print('- ', method)
            res['method'].append(method)
            res['fold'].append(k)

            X_tr_tr, X_tr_vl, y_tr_tr, y_tr_vl = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr)
            opt = -1
            print('\t- tuning')
            if method=='RF':
                for n_estimators in params[method]['n_estimators']:
                    for max_leaf_nodes in params[method]['max_leaf_nodes']:
                        clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes)
                        clf = clf.fit(X_tr_tr, y_tr_tr)
                        print('\t\t- n_estimators: {} | max_leaf_nodes {} ->  AUC: {:.4}'.format(n_estimators, max_leaf_nodes, roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1])))
                        if opt < roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1]):
                            opt = roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1])
                            clf_opt = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes)
            elif method=='LightGBM':
                for n_estimators in params[method]['n_estimators']:
                    for num_leaves in params[method]['num_leaves']:
                        clf = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves)
                        clf = clf.fit(X_tr_tr, y_tr_tr)
                        print('\t\t- n_estimators: {} | num_leaves {} ->  AUC: {:.4}'.format(n_estimators, num_leaves, roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1])))
                        if opt < roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1]):
                            opt = roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1])
                            clf_opt = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves)
            elif method=='KernelSVM':
                for C in params[method]['C']:
                    clf = SVC(C=C, kernel='rbf', probability=True)
                    clf = clf.fit(X_tr_tr, y_tr_tr)
                    print('\t\t- C: {} ->  AUC: {:.4}'.format(C, roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1])))
                    if opt < roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1]):
                        opt = roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)[:, 1])
                        clf_opt = SVC(C=C, kernel='rbf', probability=True)
            elif method=='RuleFit':
                for C in params[method]['C']:
                    clf = RuleFitClassifier(C=1/(X_tr_vl.shape[0]*C), forest=forest)
                    clf = clf.fit(X_tr_tr, y_tr_tr)
                    print('\t\t- C: {} ->  AUC: {:.4} (gsup: {} | lsup: {:.4})'.format(C, roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)), clf.global_support_cardinality(), clf.local_support_cardinality(X_tr_vl).mean()))
                    if opt < roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)):
                        opt = roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl))
                        clf_opt = RuleFitClassifier(C=1/(X_tr.shape[0]*C), forest=forest)
            elif method=='GLRM':
                for lambda0 in params[method]['lambda0']:
                    clf = GeneralizedLinerRuleClassifier(lambda0=lambda0, lambda1=0.2*lambda0, warning=False)
                    clf = clf.fit(X_tr_tr, y_tr_tr)
                    print('\t\t- lambda0: {} | lambda1: {} ->  AUC: {:.4} (gsup: {} | lsup: {:.4})'.format(lambda0, 0.2*lambda0, roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)), clf.global_support_cardinality(), clf.local_support_cardinality(X_tr_vl).mean()))
                    if opt < roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)):
                        opt = roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl))
                        clf_opt = GeneralizedLinerRuleClassifier(lambda0=lambda0, lambda1=0.2*lambda0)
            elif method=='LIRE':
                for C_l0 in params[method]['C_l0']:
                    C_li = 2.0 * C_l0
                    clf = LocallyInterpretableRuleEnsembleClassifier(C_l0=C_l0, C_li=C_li, forest=forest, max_iter=2000)
                    clf = clf.fit(X_tr_tr, y_tr_tr)
                    print('\t\t- C_l0: {} | C_li: {} -> AUC: {:.4} (gsup: {} | lsup: {:.4})'.format(C_l0, C_li, roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)), clf.global_support_cardinality(), clf.local_support_cardinality(X_tr_vl).mean()))
                    if opt < roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl)):
                        opt = roc_auc_score(y_tr_vl, clf.predict_proba(X_tr_vl))
                        clf_opt = LocallyInterpretableRuleEnsembleClassifier(C_l0=C_l0, C_li=C_li, forest=forest, max_iter=5000)
            
            s = time()
            clf_opt = clf_opt.fit(X_tr, y_tr)
            e = time() - s

            res['accuracy'].append(accuracy_score(y_ts, clf_opt.predict(X_ts)))
            res['f1'].append(f1_score(y_ts, clf_opt.predict(X_ts)))
            res['time'].append(e)

            if method in ['RuleFit', 'GLRM', 'LIRE', 'LIREF']:
                res['auc'].append(roc_auc_score(y_ts, clf_opt.predict_proba(X_ts)))
                res['gsup'].append(clf_opt.global_support_cardinality())
                res['lsup'].append(clf_opt.local_support_cardinality(X_ts).mean())
                res['rsup'].append(clf_opt.relative_supprot(X_ts))
                res['div'].append(clf_opt.diversity(X_ts))
            else:
                res['auc'].append(roc_auc_score(y_ts, clf_opt.predict_proba(X_ts)[:, 1]))
                res['gsup'].append(-1)
                res['lsup'].append(-1)
                res['rsup'].append(-1)
                res['div'].append(-1)

            for key in keys[2:]:
                print('\t- {}: {}'.format(key, res[key][-1]))
        print()

    pd.DataFrame(res).to_csv('./res/comparison_{}.csv'.format(dataset), index=False)



params = {
    'RF': {'n_estimators': [100, 300, 500], 'max_leaf_nodes': [64, 128, 256]},
    'LightGBM': {'n_estimators': [100, 300, 500], 'num_leaves': [64, 128, 256]},
    'KernelSVM': {'C': [1e-2, 5e-2, 1e-1, 5e-1, 1e+0, 5e+0, 1e+1, 5e+1, 1e+2]},
    'RuleFit': {'C': [5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2]},
    'GLRM': {'lambda0': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]},
    'LIRE': {'C_l0': [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]},
}


for dataset in ['h', 'b', 'c', 'f', 'a']:
    run(dataset=dataset, params=params)

