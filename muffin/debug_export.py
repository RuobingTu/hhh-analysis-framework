#!/usr/bin/env python3
"""Find where the exported C++ evaluator and xgboost disagree."""
import os
import sys

import numpy as np
import ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402
from muffin_model import MuffinModel  # noqa: E402
import export_muffin_cpp as ex  # noqa: E402

tag = 'poster'
m = MuffinModel.load(os.path.join(mc.OUTDIR, 'models', 'muffin_%s_nominal' % tag))
hdr = ex.emit([('nominal', m)], tag, m.feature_names)
ROOT.gInterpreter.Declare(hdr)

dr = mc.load_region('DR', verbose=False)
X = mc.select_features(dr, m.feature_names)[:4000]
py = m.predict(X)
cpp = np.array([ROOT.muffin_weight(*[float(v) for v in row]) for row in X])
d = np.abs(cpp - py) / np.maximum(py, 1e-9)
bad = np.where(d > 1e-6)[0]
print('mismatches: %d / %d   max rel %.3e' % (bad.size, py.size, d.max()))

feat, thr, left, right, miss, off = ex.flatten(m)
for i in bad[:5]:
    x = X[i]
    print('\n--- event %d  py %.6f  cpp %.6f  (rel %.2e)' % (i, py[i], cpp[i], d[i]))
    print('    features: %s' % dict(zip(m.feature_names, x)))
    # walk every tree and report nodes where the value sits exactly on a split
    for t in range(len(off) - 1):
        n = off[t]
        while feat[n] >= 0:
            v = x[feat[n]]
            if v == np.float32(thr[n]) or abs(float(v) - thr[n]) < 1e-6 * max(1.0, abs(thr[n])):
                print('    tree %d node %d: feature %s value %.9g == split %.9g'
                      % (t, n, m.feature_names[feat[n]], v, thr[n]))
            n = left[n] if v < thr[n] else right[n]
