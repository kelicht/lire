import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class FastSparseLinearClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss='exp', tol=1e-4, C_l2=1e-4, C_l0=0.0, C_li=1e-4, fit_intercept=True, class_weight=None, 
                 random_state=None, max_iter=100, pruning=True, finetune=True, verbose=False, warm_start=True):
        self.loss = loss
        self.tol = tol
        self.C_l2 = C_l2
        self.C_l0 = C_l0
        self.C_li = C_li
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.pruning = pruning
        self.finetune = finetune
        self.verbose = verbose
        self.warm_start = warm_start

    def _logistic_loss(self, beta, X, y):
        return np.log(1 + np.exp(- y * np.dot(beta, X.T))).mean() + self.C_l2 * np.linalg.norm(beta, ord=2)**2

    def _grad_logistic_loss(self, beta, X, y):
        W = (-y * np.exp(- y * np.dot(beta, X.T))) / (1 + np.exp(- y * np.dot(beta, X.T)))
        return np.array([w * x for w, x in zip(W, X)]).mean(axis=0) + 2 * self.C_l2 * beta

    def _grad_logistic_loss_1d(self, beta, X, y, d):
        W = (-y * np.exp(- y * np.dot(beta, X.T))) / (1 + np.exp(- y * np.dot(beta, X.T)))
        return np.array([w * x_d for w, x_d in zip(W, X[:, d])]).mean() + 2 * self.C_l2 * beta[d]

    def _exponential_losses(self, beta, X, y):
        return np.exp(- y * np.dot(beta, X.T))

    def _exponential_loss(self, beta, X, y):
        return self._exponential_losses(beta, X, y).mean()

    def _supp(self, beta, tol=1e-8):
        return np.where(abs(beta)>tol)[0]

    def _L0_norm(self, beta, tol=1e-8):
        return self._supp(beta, tol=tol).shape[0]

    def _local_interpretability_regularizer(self, beta, tol=1e-8):
        if self.C_li==0.0 or self._L0_norm(beta, tol=tol)==0.0:
            return 0.0
        else:
            return self.C_li_[self._supp(beta, tol=tol)].sum() / self._L0_norm(beta, tol=tol)

    def _learning_objective(self, beta, X, y, tol=1e-8):
        if self.loss=='log':
            return self._logistic_loss(beta, X, y) + self._local_interpretability_regularizer(beta, tol=tol) + self.C_l0 * self._L0_norm(beta, tol=tol)
        else:
            return self._exponential_loss(beta, X, y) + self._local_interpretability_regularizer(beta, tol=tol) + self.C_l0 * self._L0_norm(beta, tol=tol)

    def _thresholding(self, beta, X, y, d):
        c = beta[d] - self._grad_logistic_loss_1d(beta, X, y, d) / self.lipschitz_[d]
        if abs(c) - np.sqrt((2 * (self.C_l0 + self.C_li_[d])) / self.lipschitz_[d]) >= self.tol:
            return c
        else:
            return 0

    def _is_prune(self, beta, X, y, d, g_best, g_0, gg_0):
        t = self._thresholding(beta, X, y, d)
        s_1 = t; s_2 = 2 * t; 

        beta_1 = beta.copy(); beta_2 = beta.copy(); 
        beta_1[d] = s_1; beta_2[d] = s_2; 
        g_1 = self._logistic_loss(beta_1, X, y); g_2 = self._logistic_loss(beta_2, X, y); 
        gg_1 = self._grad_logistic_loss_1d(beta_1, X, y, d); gg_2 = self._grad_logistic_loss_1d(beta_2, X, y, d); 
        
        if self.C_l2 > 0:
            if g_best <= g_0 - (gg_0 ** 2) / (4 * self.C_l2):
                return True
        if gg_0 * gg_2 < 0:
            s_3 = (s_1 + s_2) / 2
            beta_3 = beta.copy(); beta_3[d] = s_3; 
            g_3 = self._logistic_loss(beta_3, X, y); gg_3 = self._grad_logistic_loss_1d(beta_3, X, y, d); 
            if self.C_l2 > 0:
                if g_best <= g_3 - (gg_3 ** 2) / (4 * self.C_l2):
                    return True
                if gg_0 * gg_3 < 0:
                    s = (-g_1 + g_3 + gg_1 * s_1 - gg_3 * s_3 + self.C_l2 * (s_3**2 - s_1**2)) / (gg_1 - gg_3 - 2 * self.C_l2 * (s_1 - s_3))
                    lb = g_1 + gg_1 * (s - s_1) + self.C_l2 * (s - s_1)**2
                else:
                    s = (-g_3 + g_2 + gg_3 * s_3 - gg_2 * s_2 + self.C_l2 * (s_2**2 - s_3**2)) / (gg_3 - gg_2 - 2 * self.C_l2 * (s_3 - s_2))
                    lb = g_3 + gg_3 * (s - s_3) + self.C_l2 * (s - s_3)**2
            else:
                if gg_0 * gg_3 < 0:
                    lb = (gg_1 * g_3 - gg_3 * g_1 + gg_1 * gg_3 * (s_1 - s_3)) / (gg_1 - gg_3)
                else:
                    lb = (gg_3 * g_2 - gg_2 * g_3 + gg_3 * gg_2 * (s_3 - s_2)) / (gg_3 - gg_2)
            if g_best <= lb:
                return True
            else:
                return False

        if self.C_l2 > 0:
            if g_best <= g_2 - (gg_2 ** 2) / (4 * self.C_l2):
                return True
        s_1 = 2 * t; s_2 = 3 * t; 
        beta_1 = beta_2; g_1 = g_2; gg_1 = gg_2; 
        beta_2 = beta.copy(); beta_2[d] = s_2; 
        g_2 = self._logistic_loss(beta_2, X, y); gg_2 = self._grad_logistic_loss_1d(beta_2, X, y, d); 
        if gg_0 * gg_2 < 0:
            s_3 = (s_1 + s_2) / 2
            beta_3 = beta.copy(); beta_3[d] = s_3; 
            g_3 = self._logistic_loss(beta_3, X, y); gg_3 = self._grad_logistic_loss_1d(beta_3, X, y, d); 
            if self.C_l2 > 0:
                if g_best <= g_3 - (gg_3 ** 2) / (4 * self.C_l2):
                    return True
                if gg_0 * gg_3 < 0:
                    s = (-g_1 + g_3 + gg_1 * s_1 - gg_3 * s_3 + self.C_l2 * (s_3**2 - s_1**2)) / (gg_1 - gg_3 - 2 * self.C_l2 * (s_1 - s_3))
                    lb = g_1 + gg_1 * (s - s_1) + self.C_l2 * (s - s_1)**2
                else:
                    s = (-g_3 + g_2 + gg_3 * s_3 - gg_2 * s_2 + self.C_l2 * (s_2**2 - s_3**2)) / (gg_3 - gg_2 - 2 * self.C_l2 * (s_3 - s_2))
                    lb = g_3 + gg_3 * (s - s_3) + self.C_l2 * (s - s_3)**2
            else:
                if gg_0 * gg_3 < 0:
                    lb = (gg_1 * g_3 - gg_3 * g_1 + gg_1 * gg_3 * (s_1 - s_3)) / (gg_1 - gg_3)
                else:
                    lb = (gg_3 * g_2 - gg_2 * g_3 + gg_3 * gg_2 * (s_3 - s_2)) / (gg_3 - gg_2)
            if g_best <= lb:
                return True
            else:
                return False
        else:
            if self.C_l2 > 0:
                if g_best <= g_2 - (gg_2 ** 2) / (4 * self.C_l2):
                    return True
            return False

    def _updateLogistic(self, beta, X, y, d, S, S_c):
        g_best = self._logistic_loss(beta, X, y)
        beta_ = beta.copy(); beta_[d] = 0; 
        g = self._logistic_loss(beta_, X, y) 
        if g <= g_best: return beta_, True

        gg = self._grad_logistic_loss(beta_, X, y)
        for d_ins in S_c[np.argsort(abs(gg[S_c]))[::-1]]:
            if self.pruning and gg[d_ins]**2 < 2 * (self.C_l0 + self.C_li_[d]) * self.lipschitz_[d_ins]:
                continue
            if self.pruning and self._is_prune(beta_, X, y, d_ins, g_best, g, gg[d_ins]):
                continue
            beta_swap = beta_.copy()
            for i in range(100):
                t = self._thresholding(beta_swap, X, y, d_ins)
                if abs(beta_swap[d_ins] - t) < self.tol:
                    break
                else:
                    beta_swap[d_ins] = t
            g_swap = self._logistic_loss(beta_swap, X, y)
            if g_swap < g_best:
                return beta_swap, True
        return beta, False

    def _finetuneLogistic(self, beta, X, y, S):
        return beta

    def _updateExponential(self, beta, Z, d, S, S_c, ls, g, li):
        beta_ = beta.copy(); beta_[d] = 0; 
        ls_ = ls * np.exp(beta[d] * Z[d])
        ls_sum = ls_.sum()
        g_ = ls_.mean()
        K_ = len(S) - 1
        C_li_S_ = li * (K_ + 1) - self.C_li_[d]
        li_ = C_li_S_ / K_ if K_>0 else 0

        D_ = ls_[Z[d]==-1].sum() / ls_sum 
        if D_ < 1e-10: D_ = 1e-10
        if D_ > 1 - 1e-10: D_ = 1 - 1e-10
        g = g_ * 2 * np.sqrt(D_ * (1-D_))
        beta[d] = 0.5 * np.log((1-D_) / D_)
        if g_ + li_ <= g + self.C_l0 + li:
            return beta_, True, ls_, g_, li_

        beta_better = beta.copy()
        ls_better = ls_ * np.exp(-1 * beta[d] * Z[d])
        g_better = g
        li_better = li
        obj_better = g_better + self.C_li_[d] / (K_+1)
        d_better = -1

        for d_ins in S_c:
            D_ = ls_[Z[d_ins]==-1].sum() / ls_sum 
            if D_ < 1e-10: D_ = 1e-10
            if D_ > 1 - 1e-10: D_ = 1 - 1e-10
            C = self.C_l0 + (self.C_li_[d_ins] - li_) / (K_+1)
            if C * (2 * g_ - C) < 0: 
                continue 
            B_ = (np.sqrt( C * (2 * g_ - C) )) / (2 * g_)
            if (D_ <= 0.5 + B_) and (D_ >= 0.5 - B_): 
                continue

            beta_swap = beta_.copy()
            beta_swap[d_ins] = 0.5 * np.log((1-D_) / D_)
            g_swap = g_ * 2 * np.sqrt(D_ * (1-D_))
            obj_swap = g_swap + self.C_li_[d_ins] / (K_+1)

            if obj_swap < obj_better:
                d_better = d_ins
                beta_better = beta_swap
                ls_better = ls_ * np.exp(-1 * beta_better[d_ins] * Z[d_ins])
                g_better = g_swap
                li_better = (C_li_S_ + self.C_li_[d_ins]) / (K_+1)
                obj_better = obj_swap

                if self.finetune: break

        return beta_better, (d_better!=-1), ls_better, g_better, li_better

    def _finetuneExponential(self, beta, Z, ls, max_iter=10):
        S = self._supp(beta)
        loss = ls.mean()
        for t in range(max_iter):
            for d in S:
                ls_ = ls * np.exp(beta[d] * Z[d])
                D_ = ls_[Z[d]==-1].sum() / ls_.sum() 
                if D_ < 1e-10: D_ = 1e-10
                if D_ > 1 - 1e-10: D_ = 1 - 1e-10
                beta[d] = 0.5 * np.log((1-D_) / D_)
                ls = ls_ * np.exp(-1 * beta[d] * Z[d])
            loss_tmp = ls.mean()
            if abs(loss - loss_tmp) < self.tol:
                break
        return beta, ls

    def getInitialSolution(self, X, y):
        if self.warm_start:
            if self.loss=='log':
                C = 1 / ((self.C_l0 + 2 * self.C_l2) * X.shape[0])
                l1_ratio = self.C_l0 / (self.C_l0 + 2 * self.C_l2)
                clf = LogisticRegression(penalty='elasticnet', C=C, solver='saga', l1_ratio=l1_ratio, fit_intercept=False, max_iter=100)
            else:
                C = 1 / (self.C_l0 * X.shape[0])
                clf = LogisticRegression(penalty='l1', C=C, solver='liblinear', fit_intercept=False, max_iter=100)
            clf = clf.fit(X, y)
            beta_0 = clf.coef_[0]
        else:
            beta_0 = np.zeros(X.shape[1])
            beta_0[np.random.choice(X.shape[1], int(X.shape[1]/2), replace=False)] = np.random.randn(int(X.shape[1]/2))
        return beta_0

    def fit(self, X, y, sample_weight=None):

        if self.fit_intercept:
            X = np.vstack([X.T, np.ones(X.shape[0])]).T
        if not (np.unique(y) == np.array([-1, 1])).all():
            y = 2 * y - 1
        if not (np.array_equal(X, X.astype(bool))):
            self.C_li = 0.0
            self.loss = 'log'

        N, D = X.shape
        self.C_li_ = self.C_li * X.mean(axis=0)
        n_fail = np.zeros(D)

        if self.loss=='log':
            self.lipschitz_ = (X**2).mean(axis=0) / 4 + 2 * self.C_l2
        if self.loss=='exp':
            H = 2 * np.array([(y==X[:, d]).mean()>0.5 for d in range(D)]) - 1
            X = (2 * X - 1) * H
            Z = X.T * y

        beta = self.getInitialSolution(X, y)
        if self.verbose:
            print('- Initial Solution')
            print('\t- Learning Obj.:', self._learning_objective(beta, X, y))
            print('\t- Learned Supp.: {} / {}'.format(self._supp(beta).shape[0], X.shape[1]))

        if self.loss=='exp':
            ls = self._exponential_losses(beta, X, y)
            g = ls.mean()
            li = self._local_interpretability_regularizer(beta)

        for t in range(self.max_iter):
            if self.verbose: print('- Iteration ', t+1)

            S = self._supp(beta) 
            S_c = np.setdiff1d(np.arange(D), S)

            is_converged = True
            for d_del in S[np.argsort(n_fail[S])]:
                if self.loss=='log':
                    beta, is_update = self._updateLogistic(beta, X, y, d_del, S, S_c)
                else:
                    beta, is_update, ls, g, li = self._updateExponential(beta, Z, d_del, S, S_c, ls, g, li)
                if is_update: 
                    is_converged = False
                    break
                else:
                    n_fail[d_del] += 1

            if is_converged: 
                if self.verbose: print('\t- Converged')
                break

            if self.finetune:
                if self.loss=='log':
                    beta = self._finetuneLogistic(beta, X, y)
                else:
                    beta, ls = self._finetuneExponential(beta, Z, ls)

            if self.verbose:
                print('\t- Learning Obj.:', self._learning_objective(beta, X, y))
                print('\t- Learned Supp.: {} / {}'.format(self._supp(beta).shape[0], X.shape[1]))

            t += 1

        if (not is_converged) and self.verbose: print('- Reached to max_iter')

        if self.loss=='log':
            beta = self._finetuneLogistic(beta, X, y)
        else:
            beta, ls = self._finetuneExponential(beta, Z, ls)

        if self.loss=='exp': 
            beta = beta * H
            beta[-1] = beta[-1] - np.sum(beta[:-1])
            beta[:-1] = 2 * beta[:-1]
        self.coef_ = beta[:-1].reshape(1,-1)
        self.intercept_ = beta[-1].reshape(-1)
        return self

    def decision_function(self, X):
        if self.loss=='log':
            return np.dot(self.coef_[0], X.T) + self.intercept_[0]
        else:
            return 2 * (np.dot(self.coef_[0], X.T) + self.intercept_[0])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-1 * self.decision_function(X)))
    
    def support(self):
        return self._supp(self.coef_)

# class FastSparseLinearClassifier



