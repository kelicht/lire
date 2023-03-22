import numpy as np
import pandas as pd
from rule_ensemble.rulefit import RuleFitClassifier
from aix360.algorithms.rbm import FeatureBinarizer
from aix360.algorithms.rbm import LogisticRuleRegression



class GeneralizedLinerRuleClassifier(RuleFitClassifier):
    def __init__(self, lambda0=0.05, lambda1=0.01, useOrd=False, debias=True, init0=False, 
                 K=1, iterMax=200, B=1, wLB=0.5, stopEarly=False, eps=1e-06, maxSolverIter=1000,
                 colCateg=[], numThresh=9, negations=True, threshStr=False, returnOrd=False,
                 warning=True):
        
        self.linear_classifier = LogisticRuleRegression(lambda0, lambda1, useOrd, debias, init0, K, iterMax, B, wLB, stopEarly, eps, maxSolverIter)
        self.binarizer = FeatureBinarizer(colCateg, numThresh, negations, threshStr, returnOrd)

        if not warning:
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    
    def _rule_to_string(self, rule):
        return rule['rule_str']

    def _get_df(self, X):
        return pd.DataFrame(X, columns=self.feature_names)

    def fit(self, X, y, sample_weight=None, 
            feature_names=[], feature_types=[], class_names=[], target_class=1):

        if np.unique(y).shape[0]!=2: y = (y == target_class).astype(int)
        self.feature_types = feature_types if len(feature_types)==X.shape[1] else ['C' for d in range(X.shape[1])]
        self.feature_names = feature_names if len(feature_names)==X.shape[1] else ['x_{}'.format(d) for d in range(X.shape[1])]
        self.class_names = class_names if len(class_names)==2 else ['Good', 'Bad']
        self.target_class = target_class

        X_df = self._get_df(X)
        X_bin = self.binarizer.fit_transform(X_df)
        self.linear_classifier.fit(X_bin, y)
        self.coef_ = self.linear_classifier.lr.coef_[0]
        self.intercept_ = self.linear_classifier.lr.intercept_[0]

        self.rules_ = []
        rules = self.linear_classifier.explain().rule.values.tolist()[1:]
        idxSort = np.abs(self.coef_).argsort()[::-1]
        X_bin = (self.linear_classifier.compute_conjunctions(X_bin) > 0).astype(int).values
        for m, rule_str, w, f in zip(idxSort, rules, self.coef_[idxSort], X_bin.mean(axis=0)[idxSort]):
            rule = {}
            rule['rule_str'] = rule_str
            rule['weight'] = w
            rule['frequency'] = f
            rule['importance'] = abs(w) * np.sqrt(f * (1-f))
            rule['index'] = m
            self.rules_.append(rule)

        self.n_rules_ = len(self.rules_)
        return self

    def transform(self, X):
        X_bin = self._transform_bin(X)
        return (self.linear_classifier.compute_conjunctions(X_bin) > 0).astype(int).values

    def _transform_bin(self, X):
        X_df = self._get_df(X)
        return self.binarizer.transform(X_df)

    def predict(self, X):
        X_bin = self._transform_bin(X)
        return self.linear_classifier.predict(X_bin)

    def predict_proba(self, X):
        X_bin = self._transform_bin(X)
        y_prob = self.linear_classifier.predict_proba(X_bin)
        if len(y_prob.shape)>1: y_prob = y_prob[:, 1]
        return y_prob

    def decision_function(self, X):
        X_bin = self._transform_bin(X)
        return self.linear_classifier.decision_function(X_bin)

    def score(self, X, y):
        X_bin = self._transform_bin(X)
        return self.linear_classifier.score(X_bin, y)

