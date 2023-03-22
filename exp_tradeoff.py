import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from time import time

from sklearn.ensemble import RandomForestClassifier
from rule_ensemble import RuleFitClassifier, LocallyInterpretableRuleEnsembleClassifier
from datasets import Dataset




def run(dataset='a', C_l1s=[1.0], C_l0s=[1.0], C_lis=[1.0], max_depth=2, n_estimators=3, n_splits=10):
    np.random.seed(0)
    D = Dataset(dataset=dataset)
    print('#', D.dataset_fullname, 'dataset')
    print()

    keys = ['method', 'fold', 'accuracy', 'f1', 'auc', 'gsup', 'lsup', 'rsup', 'div', 'C_l1', 'C_l0', 'C_li', 'time']
    res = dict( [(key, []) for key in keys] )

    k = 0
    for tr, ts in StratifiedKFold(n_splits=n_splits).split(D.X, D.y):
        k = k + 1
        X_tr, X_ts, y_tr, y_ts = D.X[tr], D.X[ts], D.y[tr], D.y[ts]
 
        print('## Fold =', k)
        print()

        print('### Random Forest')
        s = time()
        forest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators).fit(X_tr, y_tr)
        e = time() - s
        res['method'].append('RF')
        res['fold'].append(k)
        res['accuracy'].append(accuracy_score(y_ts, forest.predict(X_ts)))
        res['f1'].append(f1_score(y_ts, forest.predict(X_ts)))
        res['auc'].append(roc_auc_score(y_ts, forest.predict_proba(X_ts)[:, 1]))
        res['gsup'].append(-1)
        res['lsup'].append(-1)
        res['rsup'].append(-1)
        res['div'].append(-1)
        res['C_l1'].append(-1)
        res['C_l0'].append(-1)
        res['C_li'].append(-1)
        res['time'].append(e)
        print('- Accuracy:', forest.score(X_ts, y_ts))
        print('- F1 score:', f1_score(y_ts, forest.predict(X_ts)))
        print('- AUC     :', roc_auc_score(y_ts, forest.predict_proba(X_ts)[:, 1]))
        print('- Time [s]:', e)
        print()

        auc_opt = 0.0
        print('### RuleFit')
        for C_l1 in C_l1s:
            print('- C_l1:', C_l1)
            s = time()
            rufi = RuleFitClassifier(C=(1/X_tr.shape[0])*(1/C_l1), forest=forest)
            rufi = rufi.fit(X_tr, y_tr, feature_names=D.feature_names, feature_types=D.feature_types, class_names=D.class_names)
            e = time() - s
            res['method'].append('RuleFit')
            res['fold'].append(k)
            res['accuracy'].append(accuracy_score(y_ts, rufi.predict(X_ts)))
            res['f1'].append(f1_score(y_ts, rufi.predict(X_ts)))
            res['auc'].append(roc_auc_score(y_ts, rufi.predict_proba(X_ts)))
            res['gsup'].append(rufi.global_support_cardinality())
            res['lsup'].append(rufi.local_support_cardinality(X_ts).mean())
            res['rsup'].append(rufi.relative_supprot(X_ts))
            res['div'].append(rufi.diversity(X_ts))
            res['C_l1'].append(C_l1)
            res['C_l0'].append(-1)
            res['C_li'].append(-1)
            res['time'].append(e)
            print('\t- Accuracy:', rufi.score(X_ts, y_ts))
            print('\t- F1 score:', f1_score(y_ts, rufi.predict(X_ts)))
            print('\t- AUC     :', roc_auc_score(y_ts, rufi.predict_proba(X_ts)))
            print('\t- Time [s]:', e)
            print('\t- Rule Candidates :', rufi.coef_.shape[0])
            print('\t- Global Support  :', rufi.global_support_cardinality())
            print('\t- Local Support   :', rufi.local_support_cardinality(X_ts).mean())
            print('\t- Relative Support:', rufi.relative_supprot(X_ts))
            print('\t- Rule Diversity  :', rufi.diversity(X_ts))
            if auc_opt < roc_auc_score(y_ts, rufi.predict_proba(X_ts)):
                exp = rufi.explain_global()
                auc_opt = roc_auc_score(y_ts, rufi.predict_proba(X_ts))
        print()
        print('#### Best Learned Model')
        print(exp)
        print()

        auc_opt = 0.0
        print('### LIRE')
        for C_l0 in C_l0s:
            print('- C_l0:', C_l0)
            for C_li in C_lis:
                print('\t- C_li:', C_li)
                s = time()
                lire = LocallyInterpretableRuleEnsembleClassifier(C_l0=C_l0, C_li=C_li, forest=forest)
                lire = lire.fit(X_tr, y_tr, feature_names=D.feature_names, feature_types=D.feature_types, class_names=D.class_names)
                e = time() - s
                res['method'].append('LIRE')
                res['fold'].append(k)
                res['accuracy'].append(accuracy_score(y_ts, lire.predict(X_ts)))
                res['f1'].append(f1_score(y_ts, lire.predict(X_ts)))
                res['auc'].append(roc_auc_score(y_ts, lire.predict_proba(X_ts)))
                res['gsup'].append(lire.global_support_cardinality())
                res['lsup'].append(lire.local_support_cardinality(X_ts).mean())
                res['rsup'].append(lire.relative_supprot(X_ts))
                res['div'].append(lire.diversity(X_ts))
                res['C_l1'].append(-1)
                res['C_l0'].append(C_l0)
                res['C_li'].append(C_li)
                res['time'].append(e)
                print('\t\t- Accuracy:', lire.score(X_ts, y_ts))
                print('\t\t- F1 score:', f1_score(y_ts, lire.predict(X_ts)))
                print('\t\t- AUC     :', roc_auc_score(y_ts, lire.predict_proba(X_ts)))
                print('\t\t- Time [s]:', e)
                print('\t\t- Rule Candidates :', lire.coef_.shape[0])
                print('\t\t- Global Support  :', lire.global_support_cardinality())
                print('\t\t- Local Support   :', lire.local_support_cardinality(X_ts).mean())
                print('\t\t- Relative Support:', lire.relative_supprot(X_ts))
                print('\t\t- Rule Diversity  :', lire.diversity(X_ts))
                if auc_opt < roc_auc_score(y_ts, lire.predict_proba(X_ts)):
                    exp = lire.explain_global()
                    auc_opt = roc_auc_score(y_ts, lire.predict_proba(X_ts))
        print()
        print('#### Best Learned Model')
        print(exp)
        print()

    pd.DataFrame(res).to_csv('./res/tradeoff_{}.csv'.format(dataset), index=False)



C_l1s = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
C_l0s = [5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3]
C_lis = [1e-1, 5e-1, 1e+0, 5e+0, 10e+0]

run(dataset='a', C_l1s=C_l1s, C_l0s=C_l0s, C_lis=C_lis, max_depth=3, n_estimators=100, n_splits=10)
