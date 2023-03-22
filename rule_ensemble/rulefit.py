import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import pairwise_distances
from rule_ensemble.rule_extraction import RuleExtractor


class RuleFitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, linear_classifier=None, loss='log', sgd=False, 
                 C=1.0, class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                 forest=None, rule_type='RF', max_rule_length=3, n_estimators=100):

        if linear_classifier is None:
            if sgd:
                self.linear_classifier = SGDClassifier(loss=loss, penalty='l1', alpha=C, class_weight=class_weight, random_state=random_state, max_iter=max_iter)
            elif loss=='log':
                self.linear_classifier = LogisticRegression(penalty='l1', C=C, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter)
            elif loss=='hinge':
                self.linear_classifier = LinearSVC(penalty='l1', C=C, class_weight=class_weight, random_state=random_state, max_iter=max_iter)
        else:
            self.linear_classifier = linear_classifier

        self.rule_extractor = RuleExtractor(forest, rule_type, max_rule_length, n_estimators)

    def _rule_to_string(self, rule):
        s = ''
        for i, (d, b, t) in enumerate(zip(rule['features'], rule['branch'], rule['thresholds'])):
            if i!=0:
                s += ' & '
            if self.feature_types[d]=='B':
                if ':' in self.feature_names[d]:
                    feature, category = self.feature_names[d].split(':')
                    s += '{} {} {}'.format(feature, '!=' if b else '=', category)
                else:
                    s += '{} = {}'.format(self.feature_names[d], 'False' if b else 'True')
            elif self.feature_types[d]=='I':
                s += '{} {} {}'.format(self.feature_names[d], '<=' if b else '>', int(t))
            else:
                s += '{} {} {:.4}'.format(self.feature_names[d], '<=' if b else '>', t)
        return s

    def fit(self, X, y, sample_weight=None, 
            feature_names=[], feature_types=[], class_names=[], target_class=1):

        if np.unique(y).shape[0]!=2: y = (y == target_class).astype(int)
        self.feature_types = feature_types if len(feature_types)==X.shape[1] else ['C' for d in range(X.shape[1])]
        self.feature_names = feature_names if len(feature_names)==X.shape[1] else ['x_{}'.format(d) for d in range(X.shape[1])]
        self.class_names = class_names if len(class_names)==2 else ['Good', 'Bad']
        self.target_class = target_class

        X_bin = self.rule_extractor.fit_transform(X, y)
        self.rule_candidates_ = self.rule_extractor.rule_candidates_
        self.n_rule_candidates_ = self.rule_extractor.n_rule_candidates_

        self.linear_classifier = self.linear_classifier.fit(X_bin, y, sample_weight=sample_weight)
        self.coef_ = self.linear_classifier.coef_[0]
        self.intercept_ = self.linear_classifier.intercept_[0]

        for m, (rule, w, f) in enumerate(zip(self.rule_candidates_, self.coef_, X_bin.mean(axis=0))):
            rule['weight'] = w
            rule['frequency'] = f
            rule['importance'] = abs(w) * np.sqrt(f * (1-f))
            rule['index'] = m

        self.rules_ = [self.rule_candidates_[m] for m in self.support()]
        self.n_rules_ = len(self.rules_)
        return self

    def transform(self, X):
        return self.rule_extractor.transform(X)

    def predict(self, X):
        X_bin = self.transform(X)
        return self.linear_classifier.predict(X_bin)

    def predict_proba(self, X):
        X_bin = self.transform(X)
        y_prob = self.linear_classifier.predict_proba(X_bin)
        if len(y_prob.shape)>1: y_prob = y_prob[:, 1]
        return y_prob

    def decision_function(self, X):
        X_bin = self.transform(X)
        return self.linear_classifier.decision_function(X_bin)

    def score(self, X, y):
        X_bin = self.transform(X)
        return self.linear_classifier.score(X_bin, y)

    def support(self, tol=1e-8):
        return np.where(abs(self.coef_)>tol)[0]

    def local_support(self, X, tol=1e-8):
        X_bin = self.transform(X)
        return X_bin[:, self.support(tol=tol)]

    def global_support_cardinality(self, tol=1e-8):
        return len(self.support(tol=tol))

    def local_support_cardinality(self, X, tol=1e-8):
        return self.local_support(X, tol=tol).sum(axis=1)

    def relative_supprot(self, X, tol=1e-8):
        gsup = self.global_support_cardinality(tol=tol)
        if gsup==0: return 0
        lsup = self.local_support_cardinality(X, tol=tol).mean()
        return lsup / gsup

    def explain_global(self, sort_key='weight'):
        rules = sorted(self.rules_, key=lambda x: abs(x[sort_key]), reverse=True)
        exp = [ [rule['index'], self._rule_to_string(rule), rule['weight'], rule['frequency'], rule['importance']] for rule in rules ]
        if abs(self.intercept_)>1e-8: exp += [ [len(exp)+1, 'Intercept', self.intercept_, 1.0, 0.0] ]
        return pd.DataFrame(exp, columns=['No.', 'Rule', 'Contribution to \"{}\"'.format(self.class_names[self.target_class]), 'Frequency', 'Importance'])

    def explain_local(self, X):
        if len(X.shape)==1 and X.shape[0]==len(self.feature_names): X = X.reshape(1, -1)
        exps = []
        X_bin = self.transform(X)
        y_pred = self.predict(X)
        for x_bin, y in zip(X_bin, y_pred):
            rules = sorted([rule for rule in self.rules_ if x_bin[rule['index']]==1], key=lambda x: abs(x['weight']), reverse=True)
            exp = [ [rule['index'], self._rule_to_string(rule), rule['weight'] if y==self.target_class else -1 * rule['weight']] for rule in rules ]
            if abs(self.intercept_)>1e-8: exp += [ [len(exp)+1, 'Intercept', self.intercept_ if y==self.target_class else -1 * self.intercept_] ]
            exps.append(pd.DataFrame(exp, columns=['No.', 'Rule', 'Contribution to \"{}\"'.format(self.class_names[y])]))
        return exps

    def diversity(self, X, metric='jaccard'):
        X_bin = self.transform(X)
        if isinstance(X_bin, pd.DataFrame): X_bin = X_bin.values
        sup = self.support()
        if sup.shape[0] < 2: return 0
        X_bin = X_bin[:, sup].T
        M = 2 * X_bin.shape[0] * (X_bin.shape[0] - 1)
        return pairwise_distances(X_bin.astype(bool), metric=metric).sum() / M

# class RuleFitClassifier