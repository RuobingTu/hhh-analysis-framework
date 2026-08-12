#!/usr/bin/env python3
"""Which input features should MUFFIN use here -- the poster's list verbatim, or
the poster's list plus the axes the settled v29pre map is built on?

Trains one model per feature set on the DR and scores each in the AR, next to
the binned fr2d map evaluated on the same events.  Two numbers per set:

  AR      integral closure, predicted / observed net pass yield
  diff    yield-weighted  sqrt((1-r)^2 + sigma_r^2)  over the bins of the
          variables below -- the poster's figure-2 metric, and the one that
          actually discriminates: a model too smooth to be wrong inclusively
          still fails differentially.

Also prints the out-of-fold DR self-closure, which separates "the model is bad"
from "the DR->AR extrapolation is hard".

    bash muffin/run.sh compare_features.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402
from muffin_model import MuffinModel  # noqa: E402
from closure_muffin import load_binned_ff, binned_ff  # noqa: E402

SETS = ['poster', 'poster_abseta', 'poster_nophi', 'poster_nobtag', 'binlike']
KFOLD = int(os.environ.get('MUFFIN_KFOLD', '4'))

DIFF_VARS = [
    ('tau1Pt', np.array([20, 30, 40, 50, 65, 85, 120, 250.])),
    ('tau1jetPt', np.array([20, 34, 41, 50, 70, 100, 150, 300.])),
    ('abs_tau1Eta', np.array([0, 0.5, 0.9, 1.3, 1.7, 2.4])),
    ('nsmalljets', np.array([3.5, 4.5, 5.5, 6.5, 7.5, 12.5])),
]


def closure(w_pred, y, wsig, mask=None):
    m = np.ones_like(y, bool) if mask is None else mask
    f, p = m & (y == 0), m & (y == 1)
    return (w_pred[f] * wsig[f]).sum() / max(wsig[p].sum(), 1e-9)


def diff_metric(w_pred, y, wsig, res, per_var=False):
    out = {}
    for var, edges in DIFF_VARS:
        v, rows = res[var], []
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
            out[var] = float((a[:, 1] * a[:, 0]).sum() / a[:, 0].sum())
    return out if per_var else float(np.mean(list(out.values())))


def main():
    dr = mc.load_region('DR', verbose=False)
    ar = mc.load_region('AR', verbose=False)
    ydr, wdr = dr['y'], dr['w']
    yar, war = ar['y'], ar['w']
    rng = np.random.default_rng(1234)
    fold = rng.integers(0, KFOLD, size=ydr.size)

    # baseline: the settled binned map on the same AR events
    maps, yvar = load_binned_ff(mc.FR2D_REF)
    print('baseline map: %s  (y axis = %s)' % (os.path.basename(mc.FR2D_REF), yvar))
    w_bin = binned_ff(maps, ar['tau1jetPt'], ar[yvar],
                      ar['nsmalljets'] >= 6, ar['tau1decayMode'] >= 5)
    base_per = diff_metric(w_bin, yar, war, ar, per_var=True)
    print('%-16s %3s %7s %7s %7s | %s' % ('feature set', 'n', 'oof-DR', 'AR', 'diff',
                                          ' '.join('%-9s' % v for v, _ in DIFF_VARS)))
    print('%-16s %3s %7s %7.3f %6.1f%% | %s'
          % ('binned fr2d', '-', '-', closure(w_bin, yar, war),
             100 * np.mean(list(base_per.values())),
             ' '.join('%8.1f%%' % (100 * base_per[v]) for v, _ in DIFF_VARS)))

    for name in SETS:
        names = mc.FEATURE_SETS[name]
        Xd, Xa = mc.select_features(dr, names), mc.select_features(ar, names)
        w_cv = np.empty(ydr.size)
        for k in range(KFOLD):
            tr, te = fold != k, fold == k
            w_cv[te] = MuffinModel(names).fit(Xd[tr], ydr[tr], wdr[tr]).predict(Xd[te])
        m = MuffinModel(names).fit(Xd, ydr, wdr)
        w_ar = m.predict(Xa)
        per = diff_metric(w_ar, yar, war, ar, per_var=True)
        print('%-16s %3d %7.3f %7.3f %6.1f%% | %s'
              % (name, len(names), closure(w_cv, ydr, wdr), closure(w_ar, yar, war),
                 100 * np.mean(list(per.values())),
                 ' '.join('%8.1f%%' % (100 * per[v]) for v, _ in DIFF_VARS)))


if __name__ == '__main__':
    main()
