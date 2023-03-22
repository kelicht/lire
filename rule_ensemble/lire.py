from rule_ensemble.rulefit import RuleFitClassifier
from rule_ensemble.fslc import FastSparseLinearClassifier


class LocallyInterpretableRuleEnsembleClassifier(RuleFitClassifier):
    def __init__(self, loss='exp', tol=1e-4, C_l2=1e-4, C_l0=1e-2, C_li=1e-2, fit_intercept=True, class_weight=None, 
                 random_state=None, max_iter=100, pruning=True, finetune=True, verbose=False, warm_start=True,
                 forest=None, rule_type='RF', max_rule_length=3, n_estimators=100):

        fslc = FastSparseLinearClassifier(loss, tol, C_l2, C_l0, C_li, fit_intercept, class_weight, random_state, max_iter, pruning, finetune, verbose, warm_start)
        super().__init__(linear_classifier=fslc, forest=forest, rule_type=rule_type, max_rule_length=max_rule_length, n_estimators=n_estimators)

# class LocallyInterpretableRuleEnsembleClassifier



