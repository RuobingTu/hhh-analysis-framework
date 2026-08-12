#!/usr/bin/env python3
"""The MUFFIN density-ratio estimator (signed-weight boosted classifier).

Target (LHCP poster, CMS-PAS-TAU-25-001):

    w(z) = [p_pass^data(z) - p_pass^sim(z)] / [p_fail^data(z) - p_fail^sim(z)]
         = A(z) / B(z)

with A, B the NET (data minus subtracted genuine-tau simulation) unnormalised
densities.  A binary classifier trained on the weighted log-loss

    L = - sum_i w_i [ y_i log s_i + (1-y_i) log(1-s_i) ],    w_i signed:
        +1            for data
        -w_analysis   for genuine-tau simulation

has the per-z stationary point  s(z) = A(z) / (A(z) + B(z)), so

    w_MUFFIN(z) = s/(1-s) = exp(raw margin)

is exactly the quantity above -- no normalisation constant, because the signed
weights already carry the absolute yields.  Working with the raw margin rather
than s/(1-s) is both exact (for a logistic objective the margin IS the log
density ratio) and safer in the saturated tails.

Signed weights are essential here and not a convenience: ~15% of the pass-region
simulation carries a negative generator weight (amcatnlo/powheg-openloops), so
the genuine-tau subtraction is a cancellation between two large numbers, not a
small correction.  Estimating it as a separate density ratio (the two-step
scheme of arXiv:2511.06972) is numerically unstable at this sample size -- it
gave a 6% held-out non-closure, versus sub-percent here.

XGBoost rejects negative DMatrix weights, so the signed weights are applied
inside a custom objective instead:

    grad_i = w_i (s_i - y_i)          exact
    hess_i = |w_i| s_i (1 - s_i)      positive surrogate (damped Newton)

Using |w| in the Hessian only rescales the step; the stationary point -- where
the signed gradients cancel -- is unchanged, and tree building stays well posed
(min_child_weight keeps acting on a positive quantity).

The initial margin is log(inclusive F_F), so an untrained model returns the
inclusive fake factor and boosting only learns its z dependence: where
statistics are thin the estimator falls back to the inclusive value, not to 1.

One calibration constant is applied on top.  A perfectly converged model closes
by construction (per leaf, s/(1-s) = W_pass/W_fail, so summing w over the fail
events reproduces the net pass yield exactly), but a regularised, shrunk model
lives in log space and exp() of a shrunk margin is biased low by Jensen -- ~3%
here.  `norm` restores the determination-region normalisation, which is the one
number the DR is there to fix in the first place; it leaves all shape
information to the BDT.
"""
import json
import os

import numpy as np
import xgboost as xgb

# Chosen with scan_muffin.py: the determination region holds O(30k) data events,
# and above this capacity the differential closure stops improving while the
# in-sample closure drifts away from 1 -- i.e. the extra trees buy overfitting,
# not resolution.  min_child_weight is large because the signed weights make
# leaves with little net content very noisy.
NOMINAL = dict(max_depth=2, eta=0.05, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=10.0, min_child_weight=300.0, num_boost_round=300)

# 'modelling' systematic: the poster's "training configuration variations",
# spanning the capacity range over which the scan was flat.
VARIATIONS = {
    'deep': dict(max_depth=3),
    'deeper': dict(max_depth=4),
    'shallow': dict(max_depth=1, num_boost_round=600),
    'fastlr': dict(eta=0.10, num_boost_round=150),
    'slowlr': dict(eta=0.02, num_boost_round=750),
    'loosereg': dict(min_child_weight=50.0, reg_lambda=1.0),
    'tightreg': dict(min_child_weight=1000.0),
}

W_CLIP = (1e-4, 20.0)      # guard on the per-event fake factor


def _sigmoid(m):
    return 1.0 / (1.0 + np.exp(-np.clip(m, -30, 30)))


class MuffinModel(object):
    """MUFFIN fake factor:  w(z) = exp(margin(z))."""

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self.bst = None
        self.b0 = 0.0                 # initial margin = log(inclusive F_F)
        self.norm = 1.0               # DR normalisation (see module docstring)
        self.inclusive_ff = None
        self.n_train = None

    # -- training ---------------------------------------------------------
    def fit(self, X, y, wsig, params=None, seed=0, boot=None):
        """X features, y 1=pass/0=fail, wsig SIGNED (+1 data, -w_ana simulation).

        boot: optional per-event multiplicity for a Poisson bootstrap replica."""
        p = dict(NOMINAL)
        p.update(params or {})
        nround = int(p.pop('num_boost_round'))
        w = np.asarray(wsig, dtype=np.float64)
        if boot is not None:
            w = w * boot
        keep = w != 0
        X, y, w = X[keep], np.asarray(y, dtype=np.float64)[keep], w[keep]

        net_p, net_f = w[y == 1].sum(), w[y == 0].sum()
        if net_p <= 0 or net_f <= 0:
            raise ValueError('net pass/fail yields must be positive (%g, %g)'
                             % (net_p, net_f))
        self.inclusive_ff = float(net_p / net_f)
        self.b0 = float(np.log(self.inclusive_ff))
        self.n_train = dict(n=int(y.size), n_pass=int((y == 1).sum()),
                            n_neg_weight=int((w < 0).sum()),
                            net_pass=float(net_p), net_fail=float(net_f))

        d = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        d.set_base_margin(np.full(y.size, self.b0))

        def obj(margin, _dm):
            s = _sigmoid(margin)
            grad = w * (s - y)
            hess = np.abs(w) * np.maximum(s * (1.0 - s), 1e-6)
            return grad, hess

        p.update(seed=seed, nthread=4, disable_default_eval_metric=1)
        self.bst = xgb.train(p, d, num_boost_round=nround, obj=obj)

        # calibrate the DR normalisation on the training sample itself
        self.norm = 1.0
        raw = self.predict(X)
        pred_pass = (raw[y == 0] * w[y == 0]).sum()
        self.norm = float(net_p / pred_pass) if pred_pass > 0 else 1.0
        self.n_train['norm'] = self.norm
        return self

    # -- application ------------------------------------------------------
    def margin(self, X):
        d = xgb.DMatrix(X, feature_names=self.feature_names)
        d.set_base_margin(np.full(X.shape[0], self.b0))
        return self.bst.predict(d, output_margin=True)

    def predict(self, X, clip=W_CLIP):
        """Per-event fake factor w(z) = norm * exp(margin)."""
        w = self.norm * np.exp(np.clip(self.margin(X), -20, 20))
        return np.clip(w, clip[0], clip[1])

    # -- persistence ------------------------------------------------------
    def save(self, prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        self.bst.save_model(prefix + '.json')
        with open(prefix + '_cfg.json', 'w') as fh:
            json.dump(dict(feature_names=self.feature_names, b0=self.b0,
                           norm=self.norm, inclusive_ff=self.inclusive_ff,
                           n_train=self.n_train), fh, indent=2)

    @classmethod
    def load(cls, prefix):
        with open(prefix + '_cfg.json') as fh:
            cfg = json.load(fh)
        m = cls(cfg['feature_names'])
        m.b0 = cfg['b0']
        m.norm = cfg.get('norm', 1.0)
        m.inclusive_ff = cfg.get('inclusive_ff')
        m.n_train = cfg.get('n_train')
        m.bst = xgb.Booster()
        m.bst.load_model(prefix + '.json')
        return m

    def ranking(self):
        g = self.bst.get_score(importance_type='total_gain')
        return sorted(g.items(), key=lambda kv: -kv[1])
