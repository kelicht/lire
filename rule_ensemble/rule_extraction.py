import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class RuleExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, forest=None, rule_type='RF', max_rule_length=3, n_estimators=100):
        if forest is not None:
            self.forest = forest
            if isinstance(forest, RandomForestClassifier) or isinstance(forest, RandomForestRegressor):
                self.rule_type = 'RF'
            elif isinstance(forest, ExtraTreesClassifier) or isinstance(forest, ExtraTreesRegressor):
                self.rule_type = 'ET'
            elif isinstance(forest, GradientBoostingClassifier) or isinstance(forest, GradientBoostingRegressor):
                self.rule_type = 'GB'
            self.max_rule_length = forest.max_depth
            self.n_estimators = forest.n_estimators
            self.prefitted = True
        elif rule_type in ['RF', 'ET', 'GB']:
            self.rule_type = rule_type
            if rule_type=='RF':
                self.forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_rule_length)
            elif rule_type=='ET':
                self.forest = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_rule_length)
            elif rule_type=='GB':
                self.forest = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_rule_length)
            self.max_rule_length = max_rule_length
            self.n_estimators = n_estimators
            self.prefitted = False

    def _tree_to_rules(self, tree_obj):
        rules = []
        def rec(node_idx, D, T, B):
            if tree_obj.children_left[node_idx] != tree_obj.children_right[node_idx]:
                d = tree_obj.feature[node_idx]; t = tree_obj.threshold[node_idx]; 
                D_left = D+[d]; T_left = T+[t]; B_left = B+[True]; 
                D_right = D+[d]; T_right = T+[t]; B_right = B+[False]; 
                for i in [i for i in range(len(D)) if D[i]==d]:
                    if B[i]:
                        D_left.pop(i); T_left.pop(i); B_left.pop(i); 
                    else:
                        D_right.pop(i); T_right.pop(i); B_right.pop(i); 
                rules.append( {'features': D_left, 'thresholds': T_left, 'branch': B_left} )
                rec(tree_obj.children_left[node_idx], D_left, T_left, B_left)
                rules.append( {'features': D_right, 'thresholds': T_right, 'branch': B_right} )
                rec(tree_obj.children_right[node_idx], D_right, T_right, B_right)
        rec(0, [], [], [])
        return rules

    def fit(self, X, y=None):
        if not self.prefitted: self.forest = self.forest.fit(X, y)
        rule_candidates = []
        for tree in self.forest.estimators_: 
            rule_candidates += self._tree_to_rules(tree[0].tree_ if self.rule_type=='GB' else tree.tree_)
        self.rule_candidates_ = [rule for m, rule in enumerate(rule_candidates) if rule not in rule_candidates[m+1:]]
        self.n_rule_candidates_ = len(self.rule_candidates_)
        return self

    def transform(self, X):
        X_bin = np.zeros([X.shape[0], self.n_rule_candidates_])
        for m, rule in enumerate(self.rule_candidates_):
            X_bin[:, m] = np.array([X[:, d]<=t if b else X[:, d]>t for d, t, b in zip(rule['features'], rule['thresholds'], rule['branch'])]).all(axis=0).astype(int)
        return X_bin

    def fit_transform(self, X, y=None):
        self = self.fit(X, y=y)
        return self.transform(X)

# class RuleExtractor