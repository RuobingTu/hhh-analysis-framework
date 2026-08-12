#!/usr/bin/env python3
"""Capacity scan for MUFFIN, judged out-of-fold.

Columns
  raw    in-sample DR closure BEFORE the normalisation constant.  A model that
         is calibrated on the partition it defines closes here exactly; the
         deficit measures overfitting, since a perfectly overfit model drives
         exp(margin) -> 0 on the fail events and raw -> 0.
  norm   the constant restoring it (applied in predict).
  oof    out-of-fold integral closure -- but note a CONSTANT model scores 1.000
         here trivially, so this number alone must never pick the configuration.
  diff   the figure of merit: yield-weighted  sqrt((1-r)^2 + sigma_r^2)  over
         the bins of the DIFF_VARS below, all out-of-fold.  This is the poster's
         figure-2 metric and it does punish a model that is too smooth.
  4 classes: out-of-fold integral closure per njet x prong class.

    bash muffin/run.sh scan_muffin.py
    MUFFIN_FEATURES=nophi bash muffin/run.sh scan_muffin.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402
from muffin_model import MuffinModel  # noqa: E402

# capacity ladder: the nominal 300-round depth-3 model already overfits, so the
# scan walks down in capacity as well as up
CONFIGS = [
    ('inclusive-only', dict(max_depth=1, eta=0.05, num_boost_round=1)),
    ('d2-100',  dict(max_depth=2, eta=0.05, num_boost_round=100)),
    ('d2-300',  dict(max_depth=2, eta=0.05, num_boost_round=300)),
    ('d3-100',  dict(max_depth=3, eta=0.05, num_boost_round=100)),
    ('d3-300',  dict(max_depth=3, eta=0.05, num_boost_round=300)),
    ('d3-300-mcw100', dict(max_depth=3, eta=0.05, num_boost_round=300,
                           min_child_weight=100.0)),
    ('d3-300-mcw300', dict(max_depth=3, eta=0.05, num_boost_round=300,
                           min_child_weight=300.0)),
    ('d2-300-mcw300', dict(max_depth=2, eta=0.05, num_boost_round=300,
                           min_child_weight=300.0)),
    ('d3-600-mcw300', dict(max_depth=3, eta=0.05, num_boost_round=600,
                           min_child_weight=300.0)),
    ('d4-300-mcw300', dict(max_depth=4, eta=0.05, num_boost_round=300,
                           min_child_weight=300.0)),
]
KFOLD = int(os.environ.get('MUFFIN_KFOLD', '4'))

DIFF_VARS = [
    ('tau1Pt', np.array([20, 30, 40, 50, 65, 85, 120, 250.])),
    ('tau1jetPt', np.array([20, 34, 41, 50, 70, 100, 150, 300.])),
    ('abs_tau1Eta', np.array([0, 0.5, 0.9, 1.3, 1.7, 2.4])),
]


def closure(w_pred, y, wsig, mask=None):
    m = np.ones_like(y, bool) if mask is None else mask
    f, p = m & (y == 0), m & (y == 1)
    return (w_pred[f] * wsig[f]).sum() / max(wsig[p].sum(), 1e-9)


def diff_metric(w_pred, y, wsig, res):
    """Yield-weighted sqrt((1-r)^2 + sigma_r^2), averaged over DIFF_VARS."""
    vals = []
    for var, edges in DIFF_VARS:
        v = res[var]
        rows = []
        for k in range(len(edges) - 1):
            m = (v >= edges[k]) & (v < edges[k + 1])
            f, p = m & (y == 0), m & (y == 1)
            obs = wsig[p].sum()
            if obs <= 0 or not f.any():
                continue
            pred = (w_pred[f] * wsig[f]).sum()
            r = pred / obs
            s = np.hypot(np.sqrt(((w_pred[f] * wsig[f]) ** 2).sum()) / obs,
                         r * np.sqrt((wsig[p] ** 2).sum()) / obs)
            rows.append((obs, np.hypot(1 - r, s)))
        if rows:
            a = np.array(rows)
            vals.append((a[:, 1] * a[:, 0]).sum() / a[:, 0].sum())
    return float(np.mean(vals)) if vals else np.nan


def main():
    names = mc.FEATURE_SETS[os.environ.get('MUFFIN_FEATURES', 'poster')]
    print('features: %s' % ', '.join(names))
    dr = mc.load_region('DR')
    X = mc.select_features(dr, names)
    y, w = dr['y'], dr['w']
    nj6, p3 = dr['nsmalljets'] >= 6, dr['tau1decayMode'] >= 5
    classes = [('nj45 1p', ~nj6 & ~p3), ('nj45 3p', ~nj6 & p3),
               ('nj6 1p', nj6 & ~p3), ('nj6 3p', nj6 & p3)]

    rng = np.random.default_rng(1234)
    fold = rng.integers(0, KFOLD, size=y.size)

    print('%-16s %6s %6s %6s %7s | %s' % ('config', 'raw', 'norm', 'oof', 'diff',
                                          ' '.join('%-8s' % c for c, _ in classes)))
    only = int(os.environ.get('MUFFIN_ONLY', '0')) or len(CONFIGS)
    for name, par in CONFIGS[:only]:
        m = MuffinModel(names).fit(X, y, w, params=par)
        raw = closure(m.predict(X) / m.norm, y, w)
        w_cv = np.empty(y.size)
        for k in range(KFOLD):
            tr, te = fold != k, fold == k
            w_cv[te] = MuffinModel(names).fit(X[tr], y[tr], w[tr], params=par).predict(X[te])
        per = ' '.join('%8.3f' % closure(w_cv, y, w, msk) for _, msk in classes)
        print('%-16s %6.3f %6.3f %6.3f %6.1f%% | %s'
              % (name, raw, m.norm, closure(w_cv, y, w),
                 100 * diff_metric(w_cv, y, w, dr), per))


if __name__ == '__main__':
    main()
